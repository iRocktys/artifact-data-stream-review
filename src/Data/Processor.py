import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold
from capymoa.stream import NumpyStream


class DataStreamProcessor:
    def __init__(self, logging=True, selected_features=None):
        self.logging = logging
        self.selected_features = selected_features

    def _log(self, message):
        if self.logging:
            print(message)

    def _remove_features(
        self,
        X,
        y,
        threshold_var=None,
        threshold_corr=None,
        top_n_features=None,
    ):
        initial_count = X.shape[1]
        self._log(
            f"\n--- Iniciando Processo de Seleção de Features "
            f"(Total: {initial_count}) ---"
        )

        # Remoção por variância.
        if threshold_var is not None:
            selector = VarianceThreshold(threshold=threshold_var)
            selector.fit(X)
            cols_var = X.columns[selector.get_support()]
            removed_count = initial_count - len(cols_var)
            X = X[cols_var]

            self._log(
                f"Variância: {removed_count} features removidas. "
                f"Restantes: {X.shape[1]}"
            )
        else:
            self._log("Remoção de Variância: Pular.")

        # Remoção por correlação de Pearson.
        if threshold_corr is not None:
            corr_matrix = X.corr().abs()
            upper = corr_matrix.where(
                np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            )
            to_drop = [
                column
                for column in upper.columns
                if any(upper[column] > threshold_corr)
            ]
            X = X.drop(columns=to_drop)

            self._log(
                f"Correlação (>{threshold_corr}): "
                f"{len(to_drop)} features redundantes removidas. "
                f"Restantes: {X.shape[1]}"
            )
        else:
            self._log("Remover Correlação: Pular.")

        # Random Forest importance.
        if top_n_features is not None:
            if X.shape[1] > top_n_features:
                rf = RandomForestClassifier(
                    n_estimators=50,
                    n_jobs=-1,
                    random_state=42,
                )
                rf.fit(X, y)

                importances = pd.Series(
                    rf.feature_importances_,
                    index=X.columns,
                )
                selected_feats = (
                    importances.nlargest(top_n_features).index.tolist()
                )
                X = X[selected_feats]

                self._log(
                    f"Random Forest: Top {top_n_features} selecionadas."
                )
            else:
                self._log(
                    "Random Forest: Ignorado (Features atuais <= Top N)."
                )
        else:
            self._log("Random Forest: Pular.")

        self._log(
            f"Features Finais ({X.shape[1]}) - {X.columns.tolist()}"
        )
        self._log("--- Fim do Processo de Seleção de Features ---\n")

        return X

    def _normalize_data(self, X, method="IncrementalZScore"):
        values = np.asarray(X, dtype=np.float64)

        if values.ndim != 2:
            raise ValueError(
                "A normalização exige uma matriz bidimensional."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                "Existem NaN ou infinitos antes da normalização. "
                "Verifique a imputação, inclusive colunas totalmente ausentes."
            )

        if method is None:
            self._log("Normalização: dados originais mantidos.")
            return np.array(values, copy=True)

        method_name = str(method).strip()

        batch_scalers = {
            "MinMaxScaler": MinMaxScaler,
            "StandardScaler": StandardScaler,
            "RobustScaler": RobustScaler,
        }

        if method_name in batch_scalers:
            scaler = batch_scalers[method_name]()
            normalized = scaler.fit_transform(values)

            if not np.isfinite(normalized).all():
                raise ValueError(
                    f"A normalização '{method_name}' produziu valores "
                    "não finitos."
                )

            self._log(
                f"Normalização batch: {method_name} "
                "(ajuste realizado sobre o conjunto recebido)."
            )
            return normalized

        if method_name != "IncrementalZScore":
            supported = [
                "MinMaxScaler",
                "StandardScaler",
                "RobustScaler",
                "IncrementalZScore",
                "None",
            ]
            raise ValueError(
                f"Normalização '{method}' não suportada. "
                f"Use uma destas opções: {supported}."
            )

        # Z-score por coluna usando Welford. A instância atual
        # somente atualiza o estado depois de ser transformada.
        n_features = values.shape[1]
        mean = np.zeros(n_features, dtype=np.float64)
        m2 = np.zeros(n_features, dtype=np.float64)
        normalized = np.zeros_like(values, dtype=np.float64)
        count = 0

        for row_index, row in enumerate(values):
            if count >= 2:
                variance = np.maximum(m2 / count, 0.0)
                std = np.sqrt(variance)

                np.divide(
                    row - mean,
                    std,
                    out=normalized[row_index],
                    where=std > 0.0,
                )

            count += 1
            delta = row - mean
            mean += delta / count
            m2 += delta * (row - mean)

        if not np.isfinite(normalized).all():
            raise ValueError(
                "O z-score incremental produziu valores não finitos. "
                "Verifique a magnitude dos valores de entrada."
            )

        self._log(
            "Normalização incremental: z-score cumulativo, "
            "com estatísticas anteriores à instância atual."
        )
        return normalized

    def _handle_missing_values(self, X, method="0"):
        match str(method).lower():
            case "media":
                self._log(
                    "Tratamento de Nulos: Preenchendo com a MÉDIA das colunas..."
                )
                return X.fillna(X.mean())

            case "mediana":
                self._log(
                    "Tratamento de Nulos: Preenchendo com a MEDIANA das colunas..."
                )
                return X.fillna(X.median())

            case "moda":
                self._log(
                    "Tratamento de Nulos: Preenchendo com a MODA das colunas..."
                )
                return X.fillna(X.mode().iloc[0])

            case "0":
                self._log("Tratamento de Nulos: Preenchendo com ZERO.")
                return X.fillna(0)

            case _:
                self._log(
                    f"Aviso: Método de preenchimento '{method}' desconhecido. "
                    "Usando ZERO por padrão."
                )
                return X.fillna(0)

    def _encode_labels(self, y_series, binary_label):
        y_str = y_series.astype(str).str.strip()

        if binary_label:
            self._log(
                "Target: Binarizando rótulos (0=BENIGN, 1=ATTACK)..."
            )
            is_benign = y_str.str.upper() == "BENIGN"
            y = np.where(is_benign, 0, 1).astype(np.int8)
            target_names = ["BENIGN", "ATTACK"]

        else:
            self._log(
                "Target: Mantendo multiclasse (Forçando BENIGN=0)..."
            )
            unique_labels = y_str.unique().tolist()

            # Encontra o label normal e força ele a ser o índice 0.
            normal_label = next(
                (
                    label
                    for label in unique_labels
                    if label.upper() in ["BENIGN", "NORMAL"]
                ),
                None,
            )

            if normal_label and normal_label in unique_labels:
                unique_labels.remove(normal_label)
                unique_labels.insert(0, normal_label)

            mapping = {
                label: idx
                for idx, label in enumerate(unique_labels)
            }
            y = y_str.map(mapping).fillna(-1).astype(np.int8)
            target_names = unique_labels

        return y, target_names

    def create_stream(
        self,
        df,
        target_label_col="Label",
        binary_label=True,
        normalize_method="IncrementalZScore",
        threshold_var=None,
        threshold_corr=None,
        top_n_features=None,
        return_stream=True,
        extra_ignore_cols=None,
        imputation_method="0",
    ):
        # Limpeza básica.
        self._log(
            "Limpeza: Removendo espaços, identificadores e colunas vazias..."
        )
        df = df.copy()
        df.columns = df.columns.str.strip()
        target_label_col = target_label_col.strip()

        # Filtro global de features antes do processamento.
        if self.selected_features is not None:
            self._log(
                f"Filtro Global Ativo: Mantendo apenas as "
                f"{len(self.selected_features)} features especificadas."
            )

            cols_to_keep = [
                column
                for column in self.selected_features
                if column in df.columns
            ]

            if (
                target_label_col in df.columns
                and target_label_col not in cols_to_keep
            ):
                cols_to_keep.append(target_label_col)

            df = df[cols_to_keep]

        ignore_cols = [
            "Flow ID",
            "Timestamp",
            "SimillarHTTP",
            "Unnamed: 0",
        ]

        if extra_ignore_cols:
            if isinstance(extra_ignore_cols, str):
                ignore_cols.append(extra_ignore_cols)
            else:
                ignore_cols.extend(extra_ignore_cols)

        cols_to_drop = [
            column
            for column in ignore_cols
            if column in df.columns
        ]

        X = df.drop(
            columns=[target_label_col] + cols_to_drop,
            errors="ignore",
        )

        # Tratamento numérico.
        self._log("Pré-processamento: Convertendo infinitos...")
        X = X.select_dtypes(include=[np.number]).copy()

        X.replace(
            [np.inf, -np.inf],
            [
                np.finfo(np.float32).max,
                np.finfo(np.float32).min,
            ],
            inplace=True,
        )
        X = self._handle_missing_values(
            X,
            method=imputation_method,
        )

        # Transformação na ordem das linhas que formarão a stream.
        # O NumpyStream armazena o resultado; restart() reproduz os mesmos
        # vetores, sem reutilizar estatísticas do fim da execução anterior.
        temp_x_array = self._normalize_data(
            X,
            method=normalize_method,
        )
        X = pd.DataFrame(
            temp_x_array,
            columns=X.columns,
            index=X.index,
        )

        # Definição do target e encoding.
        y, target_names = self._encode_labels(
            df[target_label_col],
            binary_label,
        )

        # Redução da dimensionalidade.
        if (
            threshold_var is not None
            or threshold_corr is not None
            or top_n_features is not None
        ):
            self._log(
                "Seleção de Features: Iniciando pipeline de redução "
                "de dimensionalidade..."
            )
            X = self._remove_features(
                X,
                y,
                threshold_var=threshold_var,
                threshold_corr=threshold_corr,
                top_n_features=top_n_features,
            )
        else:
            self._log(
                "Seleção de Features: Nenhuma técnica dinâmica selecionada. "
                "Mantendo colunas atuais."
            )

        feature_names = X.columns.tolist()
        final_x_array = X.values

        if return_stream:
            self._log(
                "Finalização: Criando objeto NumpyStream para o CapyMOA.\n"
            )
            stream_obj = NumpyStream(
                final_x_array,
                y,
                target_name="Class",
                feature_names=feature_names,
                target_type="categorical",
            )
            return stream_obj, target_names, feature_names

        self._log(
            "Finalização: Retornando DataFrame pandas processado.\n"
        )
        final_df = pd.DataFrame(
            final_x_array,
            columns=feature_names,
        )
        return final_df, y, target_names