import os
import glob

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.covariance import LedoitWolf

from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list


class DatasetHandler:
    def __init__(self, logging=True):
        self.logging = logging

    def _log(self, message):
        if self.logging:
            print(message)

    def create_balanced_dataset(
        self,
        src_dir,
        dest_dir,
        output_filename,
        n_samples_per_class,
        chunk_size=100000,
        target_files=None,
        ignored_classes=None,
        allow_insufficient=False,
    ):
        if ignored_classes is None:
            ignored_classes = []

        if not os.path.exists(dest_dir):
            os.makedirs(dest_dir)

        if target_files:
            csv_files = [
                os.path.join(src_dir, filename)
                for filename in target_files
            ]
            csv_files = [
                filename
                for filename in csv_files
                if os.path.exists(filename)
            ]
        else:
            csv_files = glob.glob(os.path.join(src_dir, "*.csv"))

        if not csv_files:
            self._log(
                f" [ERRO] Nenhum arquivo CSV válido encontrado em {src_dir}"
            )
            return

        full_output_path = os.path.join(dest_dir, output_filename)

        if os.path.exists(full_output_path):
            os.remove(full_output_path)

        self._log("=" * 80)
        self._log(f"PROCESSAMENTO OTIMIZADO (CHUNKS): {output_filename}")
        self._log(f"Arquivos Selecionados: {len(csv_files)}")
        self._log(f"Tamanho do Lote (Chunksize): {chunk_size}")

        if ignored_classes:
            self._log(f"Classes Ignoradas: {ignored_classes}")

        self._log("=" * 80)
        self._log("\n[*] Varredura global (Lendo em lotes)...")

        global_counts = {}

        for file in csv_files:
            try:
                header_df = pd.read_csv(
                    file,
                    nrows=0,
                    encoding_errors="ignore",
                    on_bad_lines="skip",
                )

                col_label = next(
                    (
                        column
                        for column in header_df.columns
                        if "label" in column.lower()
                    ),
                    None,
                )

                if not col_label:
                    continue

                for chunk in pd.read_csv(
                    file,
                    usecols=[col_label],
                    chunksize=chunk_size,
                    encoding_errors="ignore",
                    on_bad_lines="skip",
                ):
                    chunk.columns = ["Label"]
                    counts = chunk["Label"].value_counts().to_dict()

                    for label_class, quantity in counts.items():
                        if label_class not in ignored_classes:
                            global_counts[label_class] = (
                                global_counts.get(label_class, 0)
                                + quantity
                            )

            except Exception as error:
                self._log(
                    f" [!] Erro ao ler {os.path.basename(file)}: {error}"
                )

        self._log("[*] Validando quantidades disponíveis...")

        insufficient_classes = {}
        classes_to_collect = []

        for label_class, total in global_counts.items():
            if total < n_samples_per_class:
                insufficient_classes[label_class] = total
            else:
                classes_to_collect.append(label_class)

        if insufficient_classes:
            if not allow_insufficient:
                self._log("\n" + "!" * 80)
                self._log(
                    " [ERRO CRÍTICO] Classes insuficientes detectadas "
                    "(Processo Interrompido):"
                )

                for label_class, total in insufficient_classes.items():
                    self._log(
                        f"   -> {label_class}: {total} "
                        f"(Meta: {n_samples_per_class})"
                    )

                self._log("!" * 80)
                return

            insufficient_names = list(insufficient_classes.keys())

            self._log(
                f" [AVISO] Permitindo {len(insufficient_classes)} "
                f"classes insuficientes: {insufficient_names}"
            )
            classes_to_collect.extend(insufficient_classes.keys())

        if not classes_to_collect:
            self._log(" [ERRO] Nenhuma classe para coletar.")
            return

        self._log(
            f"\n[*] Coletando e Salvando em disco "
            f"(Lotes de {chunk_size})..."
        )

        collected_samples_count = {
            label_class: 0
            for label_class in classes_to_collect
        }

        buffer_list = []
        buffer_row_count = 0
        buffer_limit = chunk_size
        first_write = True

        for file in csv_files:
            all_done = True

            for label_class in classes_to_collect:
                target = min(
                    n_samples_per_class,
                    global_counts[label_class],
                )

                if collected_samples_count[label_class] < target:
                    all_done = False
                    break

            if all_done:
                break

            self._log(f"   -> Processando: {os.path.basename(file)}")

            try:
                for chunk in pd.read_csv(
                    file,
                    chunksize=chunk_size,
                    encoding_errors="ignore",
                    on_bad_lines="skip",
                ):
                    chunk.columns = chunk.columns.str.strip()

                    if "Label" not in chunk.columns:
                        continue

                    chunk_selection = []

                    for label_class in classes_to_collect:
                        target = min(
                            n_samples_per_class,
                            global_counts[label_class],
                        )
                        current = collected_samples_count[label_class]

                        if current >= target:
                            continue

                        class_chunk_df = chunk[
                            chunk["Label"] == label_class
                        ]

                        if class_chunk_df.empty:
                            continue

                        remaining = target - current
                        n_to_select = min(
                            remaining,
                            len(class_chunk_df),
                        )

                        samples = class_chunk_df.sample(
                            n=n_to_select,
                            random_state=42,
                        )

                        chunk_selection.append(samples)
                        collected_samples_count[label_class] += n_to_select

                    if chunk_selection:
                        chunk_merged = pd.concat(chunk_selection)
                        buffer_list.append(chunk_merged)
                        buffer_row_count += len(chunk_merged)

                    if buffer_row_count >= buffer_limit:
                        df_buffer = pd.concat(buffer_list)
                        mode = "w" if first_write else "a"

                        df_buffer.to_csv(
                            full_output_path,
                            mode=mode,
                            header=first_write,
                            index=False,
                        )

                        self._log(
                            f"      [IO] Buffer cheio "
                            f"({buffer_row_count} linhas). "
                            "Salvando lote no disco..."
                        )

                        buffer_list = []
                        buffer_row_count = 0
                        first_write = False

            except Exception as error:
                self._log(
                    f" [!] Erro ao processar chunk em {file}: {error}"
                )

        if buffer_list:
            df_buffer = pd.concat(buffer_list)
            mode = "w" if first_write else "a"

            df_buffer.to_csv(
                full_output_path,
                mode=mode,
                header=first_write,
                index=False,
            )

            self._log(
                f"      [IO] Salvando lote final "
                f"({len(df_buffer)} linhas)..."
            )

        self._log("\n" + "=" * 80)
        self._log("CONCLUÍDO COM SUCESSO")
        self._log(f"Arquivo gerado: {full_output_path}")
        self._log("=" * 80)

    def sort_dataset_by_timestamp(self, file_path):
        if not os.path.exists(file_path):
            self._log(f" [ERRO] Arquivo não encontrado: {file_path}")
            return

        self._log("=" * 80)
        self._log(
            "INICIANDO ORDENAÇÃO CRONOLÓGICA: "
            f"{os.path.basename(file_path)}"
        )
        self._log("=" * 80)

        try:
            df = pd.read_csv(
                file_path,
                encoding_errors="ignore",
                on_bad_lines="skip",
            )
            df.columns = df.columns.str.strip()

            timestamp_col = None

            for column in df.columns:
                if (
                    "timestamp" in column.lower()
                    or "time" in column.lower()
                    or "date" in column.lower()
                ):
                    timestamp_col = column
                    break

            if not timestamp_col:
                self._log(
                    " [ERRO] Nenhuma coluna de data/hora "
                    "(Timestamp) foi encontrada no dataset."
                )
                return

            self._log(
                f" [INFO] Coluna de tempo identificada: "
                f"'{timestamp_col}'. Convertendo dados..."
            )

            df[timestamp_col] = pd.to_datetime(
                df[timestamp_col],
                errors="coerce",
            )

            self._log(" [INFO] Ordenando as amostras...")
            df = df.sort_values(by=timestamp_col)

            self._log(
                f" [INFO] Salvando atualizações no arquivo: "
                f"{file_path}..."
            )
            df.to_csv(file_path, index=False)

            self._log("\n" + "=" * 50)
            self._log(" ORDENAÇÃO CONCLUÍDA COM SUCESSO")
            self._log("=" * 50)

            if "Label" in df.columns:
                self._log(f"Total de Amostras no Arquivo: {len(df)}")
                self._log("-" * 30)
                self._log("Contagem por Rótulo (Label):")
                self._log(df["Label"].value_counts())
            else:
                self._log(
                    " [AVISO] Coluna 'Label' não encontrada "
                    "para realizar a contagem."
                )

            self._log("=" * 50)

        except Exception as error:
            self._log(
                f" [!] Ocorreu um erro durante a ordenação: {error}"
            )

    def _validate_feature_data(self, X, y, target_names):
        if not isinstance(X, pd.DataFrame) or X.empty:
            raise ValueError("X deve ser um DataFrame não vazio.")

        if not X.columns.is_unique:
            raise ValueError("X contém nomes de features duplicados.")

        if not np.isfinite(
            X.to_numpy(dtype=np.float64)
        ).all():
            raise ValueError(
                "X contém NaN ou infinitos. "
                "Revise o pré-processamento."
            )

        y = np.asarray(y)

        if y.ndim != 1 or len(y) != len(X):
            raise ValueError(
                "y deve corresponder às linhas de X, na mesma ordem."
            )

        if not np.issubdtype(y.dtype, np.integer):
            raise ValueError(
                "y deve conter índices inteiros de target_names."
            )

        if y.min() < 0 or y.max() >= len(target_names):
            raise ValueError("Há índices de classes inválidos em y.")

        if len(np.unique(y)) < 2:
            raise ValueError(
                "São necessárias pelo menos duas classes."
            )

        names = [
            str(target_names[index])
            for index in np.unique(y)
        ]

        if len(set(names)) != len(names):
            raise ValueError(
                "Os nomes das classes devem ser únicos."
            )

        return y

    def extract_ovr_feature_importance(
        self,
        X,
        y,
        target_names,
        top_per_class=10,
    ):
        """RF One-vs-Rest por classe, incluindo BENIGN.

        Utiliza apenas dados de desenvolvimento.

        Retorna:
            class_top_features: seleção ordenada de cada classe.
            consolidated_features: união de todas as seleções.

        Não utiliza filtro por frequência entre classes.
        """
        y = self._validate_feature_data(X, y, target_names)

        if (
            not isinstance(top_per_class, (int, np.integer))
            or top_per_class < 1
        ):
            raise ValueError(
                "top_per_class deve ser um inteiro positivo."
            )

        features = X.columns.tolist()
        class_top_features = {}
        all_important_features = set()

        for cls_idx in np.unique(y):
            cls_name = str(target_names[cls_idx])

            self._log(
                f"[RF] {cls_name} vs demais classes"
            )

            rf = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1,
            )

            rf.fit(
                X,
                (y == cls_idx).astype(np.int8),
            )

            indices = np.argsort(
                -rf.feature_importances_,
                kind="stable",
            )

            selected = [
                features[index]
                for index in indices[:top_per_class]
            ]

            class_top_features[cls_name] = selected
            all_important_features.update(selected)

        # Ordem determinística, seguindo as colunas originais.
        consolidated_features = [
            feature
            for feature in features
            if feature in all_important_features
        ]

        self._log(
            f"[RF] Features finais "
            f"({len(consolidated_features)}):"
        )
        self._log(consolidated_features)

        return class_top_features, consolidated_features

    def plot_similarity_and_feature_groups(
        self,
        X,
        y,
        target_names,
        selected_features,
        metric="mahalanobis",
        linkage_method="average",
    ):
        if metric != "mahalanobis":
            raise ValueError(
                "Esta análise utiliza metric='mahalanobis'."
            )

        if linkage_method not in {
            "average",
            "complete",
            "single",
        }:
            raise ValueError(
                "Use linkage_method='average', "
                "'complete' ou 'single'."
            )

        selected_features = list(selected_features)

        if (
            not selected_features
            or len(set(selected_features)) != len(selected_features)
        ):
            raise ValueError(
                "Informe uma lista não vazia de features distintas."
            )

        missing = [
            feature
            for feature in selected_features
            if feature not in X.columns
        ]

        if missing:
            raise ValueError(
                f"Features não encontradas em X: {missing}"
            )

        selected_X = (
            X.loc[:, selected_features]
            .astype(np.float64)
            .copy()
        )

        y = self._validate_feature_data(
            selected_X,
            y,
            target_names,
        )

        # Agrupamento posicional: não depende dos índices de X ou y.
        signatures = selected_X.groupby(
            y,
            sort=True,
        ).mean()

        signatures.index = [
            str(target_names[index])
            for index in signatures.index
        ]
        signatures.index.name = "Class"

        values = selected_X.to_numpy()

        # Features constantes não diferenciam os centroides.
        # Permanecem no mapa de calor e na lista do RF,
        # mas não entram no cálculo da distância.
        variable = np.any(
            values != values[0],
            axis=0,
        )

        distance_features = (
            selected_X.columns[variable].tolist()
        )
        constant_features = (
            selected_X.columns[~variable].tolist()
        )

        if not distance_features:
            raise ValueError(
                "Todas as features selecionadas são constantes."
            )

        if constant_features:
            self._log(
                "[Mahalanobis] Constantes excluídas da distância: "
                f"{constant_features}"
            )

        # Covariância regularizada das instâncias de desenvolvimento.
        covariance_model = LedoitWolf().fit(
            values[:, variable]
        )
        precision = covariance_model.precision_

        if not np.isfinite(precision).all():
            raise ValueError(
                "A matriz de precisão contém valores não finitos."
            )

        centers = signatures.loc[
            :,
            distance_features,
        ].to_numpy()

        # VI é a inversa regularizada da covariância.
        distances = pdist(
            centers,
            metric="mahalanobis",
            VI=precision,
        )

        if not np.isfinite(distances).all():
            raise ValueError(
                "As distâncias contêm valores não finitos."
            )

        Z = linkage(
            distances,
            method=linkage_method,
            optimal_ordering=True,
        )
        order = leaves_list(Z)

        # Classes seguem o dendrograma.
        # Features mantêm a ordem da seleção.
        ordered_signatures = signatures.iloc[order]

        distance_matrix = pd.DataFrame(
            squareform(distances),
            index=signatures.index,
            columns=signatures.index,
        )

        # Dendrograma das classes.
        fig, ax = plt.subplots(
            figsize=(
                max(10, 0.8 * len(signatures)),
                6,
            )
        )

        dendrogram(
            Z,
            labels=signatures.index.tolist(),
            leaf_rotation=45,
            leaf_font_size=11,
            ax=ax,
        )

        ax.set_xlabel("Class")
        ax.set_ylabel("Mahalanobis distance")
        ax.grid(
            axis="y",
            linestyle="--",
            alpha=0.4,
        )

        fig.tight_layout()
        plt.show()
        plt.close(fig)

        # Mapa de calor das médias: sem outra normalização global.
        fig, ax = plt.subplots(
            figsize=(
                max(12, 0.45 * len(selected_features)),
                max(5, 0.5 * len(signatures) + 2),
            )
        )

        sns.heatmap(
            ordered_signatures,
            cmap="coolwarm",
            center=0,
            annot=False,
            linewidths=0.5,
            cbar_kws={
                "label": "Mean feature z-score by class",
                "shrink": 0.7,
            },
            ax=ax,
        )

        ax.set_xlabel("Selected features")
        ax.set_ylabel("Class")

        plt.setp(
            ax.get_xticklabels(),
            rotation=60,
            ha="right",
        )
        plt.setp(
            ax.get_yticklabels(),
            rotation=0,
        )

        fig.tight_layout()
        plt.show()
        plt.close(fig)

        return {
            "class_means": signatures,
            "ordered_class_means": ordered_signatures,
            "distance_matrix": distance_matrix,
            "covariance": pd.DataFrame(
                covariance_model.covariance_,
                index=distance_features,
                columns=distance_features,
            ),
            "precision": pd.DataFrame(
                precision,
                index=distance_features,
                columns=distance_features,
            ),
            "distance_features": distance_features,
            "constant_features": constant_features,
            "linkage": Z,
        }