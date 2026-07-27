from pathlib import Path
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter
from src.Results.PlotsBestModels import PlotsBestModels

class PlotPaper:
    def __init__(
        self,
        projectRoot=".",
        outputDirectory="output/Results/plots_paper",
        metadataPath="data/attack_regions_metadata.json",
        dataRoot="data/15k",
        warmUp=2000,
        windowSize=100,
        showFigure=True,
    ):
        self.projectRoot = Path(projectRoot).expanduser().resolve()
        self.outputDirectory = self.resolvePath(outputDirectory)
        self.metadataPath = self.resolvePath(metadataPath)
        self.dataRoot = self.resolvePath(dataRoot)
        self.warmUp = int(warmUp)
        self.windowSize = int(windowSize)
        self.showFigure = bool(showFigure)

        self.modelFolders = [
            "HoeffdingTree",
            "HoeffdingAdaptiveTree",
            "AdaptiveRandomForest",
            "LeveragingBagging",
            "AdaptiveIsolationForest",
            "HalfSpaceTrees",
            "Autoencoder",
        ]

        self.scenarios = [
            "Consistency_1000",
            "Generalization_1000",
            "Adaptation_200",
            "Recurrence_200",
        ]

        self.scenarioConfiguration = {
            "Consistency_1000": {
                "name": "Consistency",
                "block": "1k/block",
            },
            "Generalization_1000": {
                "name": "Generalization",
                "block": "1k/block",
            },
            "Adaptation_200": {
                "name": "Adaptation",
                "block": "200/block",
            },
            "Recurrence_200": {
                "name": "Recurrence",
                "block": "100/block",
            },
        }

        self.errorColors = {
            "FP": "#0072B2",
            "FN": "#D62728",
        }

        self.attackColors = {
            "DNS": "#E69F00",
            "SYN": "#009E73",
            "LDAP": "#CC79A7",
        }

        self.plots = None
        self.metadata = None
        self.selectedModels = None
        self.runs = {}
        self.blocksByScenario = {}
        self.maxInstance = None
        self.xLimits = None
        self.yUpperLimit = None
        self.plotYLimits = None
        self.xTicks = None
        self.yTicks = None

    def resolvePath(self, path):
        path = Path(path).expanduser()

        if path.is_absolute():
            return path

        return self.projectRoot / path

    def configureStyle(self):
        plt.rcParams.update(
            {
                "font.family": "serif",
                "font.serif": [
                    "Times New Roman",
                    "Times",
                    "DejaVu Serif",
                ],
                "font.size": 8.0,
                "axes.titlesize": 7.8,
                "xtick.labelsize": 7.0,
                "ytick.labelsize": 7.0,
                "legend.fontsize": 6.8,
                "figure.dpi": 200,
                "savefig.dpi": 400,
                "pdf.fonttype": 42,
                "ps.fonttype": 42,
            }
        )

    def findResults(self):
        allCsvFiles = {}

        for folder in self.modelFolders:
            resultsDirectory = self.projectRoot / "output" / folder

            if not resultsDirectory.exists():
                continue

            for path in resultsDirectory.rglob("*.csv"):
                pathParts = [part.lower() for part in path.parts]

                if "prequential" not in pathParts:
                    allCsvFiles[str(path)] = None

        if not allCsvFiles:
            raise FileNotFoundError(
                "No CSV files were found in "
                f"{self.projectRoot / 'output'}/<model>."
            )

        return allCsvFiles

    def loadMetadata(self):
        self.plots = PlotsBestModels(
            metadata_path=str(self.metadataPath)
        )

        self.metadata = self.plots.load_attack_metadata()

        if not self.metadata:
            self.metadata = self.plots.build_attack_metadata(
                data_root=str(self.dataRoot),
                label_col="Label",
                output_path=str(self.metadataPath),
            )

        missingScenarios = [
            scenario
            for scenario in self.scenarios
            if scenario not in self.metadata
        ]

        if missingScenarios:
            raise KeyError(
                "Missing metadata for: " + ", ".join(missingScenarios)
            )

    @staticmethod
    def groupBlocks(regions, maximumGap=500):
        blocks = []
        sortedRegions = sorted(
            regions,
            key=lambda item: item["start"],
        )

        for region in sortedRegions:
            current = {
                "start": int(region["start"]),
                "end": int(region["end"]),
                "attack_name": str(
                    region["attack_name"]
                ).upper(),
            }

            sameBlock = (
                blocks
                and blocks[-1]["attack_name"]
                == current["attack_name"]
                and current["start"] - blocks[-1]["end"] - 1
                <= maximumGap
            )

            if sameBlock:
                blocks[-1]["end"] = max(
                    blocks[-1]["end"],
                    current["end"],
                )
            else:
                blocks.append(current)

        return blocks

    def selectModels(self, allCsvFiles):
        candidates = self.plots.build_candidates_dataframe(
            allCsvFiles,
            window_size=self.windowSize,
        )

        self.selectedModels = self.plots.select_best_models(
            candidates,
            top_n=1,
        )

        self.selectedModels = self.selectedModels[
            self.selectedModels["English_Dataset"].isin(self.scenarios)
        ].copy()

        taskTypeCounts = (
            self.selectedModels
            .groupby("English_Dataset")["Task_Type"]
            .nunique()
            .reindex(self.scenarios)
            .fillna(0)
        )

        if taskTypeCounts.lt(2).any():
            missingScenarios = taskTypeCounts[
                taskTypeCounts.lt(2)
            ].index.tolist()

            raise RuntimeError(
                "A classifier and an anomaly detector were not found for: "
                + ", ".join(missingScenarios)
            )

    def loadRuns(self):
        self.runs = {}

        for scenario in self.scenarios:
            for taskType in ("classifier", "anomaly"):
                result = self.selectedModels[
                    (
                        self.selectedModels["English_Dataset"]
                        == scenario
                    )
                    & (
                        self.selectedModels["Task_Type"]
                        == taskType
                    )
                ]

                if result.empty:
                    raise RuntimeError(
                        f"Missing result for {scenario} ({taskType})."
                    )

                row = result.iloc[0]
                series = (
                    self.plots
                    ._read_prequential_series(row)
                    .sort_values("Instance")
                    .reset_index(drop=True)
                )

                if series.empty:
                    raise RuntimeError(
                        f"Empty series for {scenario} ({taskType})."
                    )

                self.runs[(scenario, taskType)] = {
                    "row": row,
                    "series": series,
                }

        self.blocksByScenario = {
            scenario: self.groupBlocks(
                self.metadata[scenario]["attack_regions"]
            )
            for scenario in self.scenarios
        }

    def calculateGlobalYLimit(self):
        maximumValues = []

        for run in self.runs.values():
            series = run["series"]
            x = series["Instance"].to_numpy(dtype=float)
            evaluationMask = x >= self.warmUp

            for metric, deviation in (
                ("FP_avg", "FP_std"),
                ("FN_avg", "FN_std"),
            ):
                mean = series[metric].to_numpy(dtype=float)

                if deviation in series.columns:
                    standardDeviation = (
                        series[deviation]
                        .fillna(0)
                        .to_numpy(dtype=float)
                    )
                else:
                    standardDeviation = np.zeros_like(mean)

                upperLimit = mean + standardDeviation
                validMask = (
                    evaluationMask
                    & np.isfinite(upperLimit)
                )

                if validMask.any():
                    maximumValues.append(
                        np.nanmax(upperLimit[validMask])
                    )

        if not maximumValues:
            return 10

        maximumValue = max(maximumValues)

        if maximumValue <= 10:
            step = 2
        elif maximumValue <= 25:
            step = 5
        elif maximumValue <= 50:
            step = 10
        else:
            step = 20

        return min(
            100,
            max(
                step,
                np.ceil(maximumValue / step) * step,
            ),
        )

    def configureAxes(self):
        self.maxInstance = max(
            np.nanmax(
                run["series"]["Instance"].to_numpy(dtype=float)
            )
            for run in self.runs.values()
        )

        self.xLimits = (
            -self.maxInstance * 0.025,
            self.maxInstance * 1.015,
        )

        self.yUpperLimit = self.calculateGlobalYLimit()
        yLowerMargin = max(
            2.5,
            self.yUpperLimit * 0.10,
        )
        self.plotYLimits = (
            -yLowerMargin,
            self.yUpperLimit,
        )
        self.yTicks = np.linspace(
            0,
            self.yUpperLimit,
            5,
        )
        self.xTicks = list(
            np.arange(
                0,
                self.maxInstance + 1,
                4000,
            )
        )

    @staticmethod
    def formatInstance(value, position):
        if np.isclose(value, 0):
            return "0"

        if abs(value) >= 1000:
            valueInThousands = value / 1000

            if float(valueInThousands).is_integer():
                return f"{int(valueInThousands)}k"

            return f"{valueInThousands:.1f}k"

        return f"{value:.0f}"

    @staticmethod
    def formatError(value, position):
        if np.isclose(value, round(value)):
            return str(int(round(value)))

        return f"{value:.1f}"

    def compactModelName(self, row):
        name = self.plots._series_label(row)

        replacements = {
            "Hoeffding Adaptive Tree": "HAT",
            "HoeffdingAdaptiveTree": "HAT",
            "Hoeffding Tree": "HT",
            "HoeffdingTree": "HT",
            "Adaptive Random Forest": "ARF",
            "AdaptiveRandomForest": "ARF",
            "Leveraging Bagging": "LB",
            "LeveragingBagging": "LB",
            "Adaptive Isolation Forest": "AIF",
            "AdaptiveIsolationForest": "AIF",
            "Half-Space Trees": "HST",
            "Half Space Trees": "HST",
            "HalfSpaceTrees": "HST",
            "Autoencoder": "AE",
            "Full Features": "All Features",
            "Full features": "All Features",
            "All features": "All Features",
            "Best Features": "Selected Features",
            "Best features": "Selected Features",
            "Selected features": "Selected Features",
            " - ": " . ",
            " · ": " . ",
            " | ": " . ",
        }

        for oldText, newText in replacements.items():
            name = name.replace(oldText, newText)

        return name

    def drawPanel(
        self,
        axis,
        scenario,
        taskType,
        panelLetter,
    ):
        run = self.runs[(scenario, taskType)]
        row = run["row"]
        series = run["series"]
        configuration = self.scenarioConfiguration[scenario]
        x = series["Instance"].to_numpy(dtype=float)
        evaluationMask = x >= self.warmUp

        axis.axvspan(
            0,
            self.warmUp,
            facecolor="#BDBDBD",
            alpha=0.35,
            linewidth=0,
            zorder=1,
        )

        for index, block in enumerate(
            self.blocksByScenario[scenario]
        ):
            start = max(
                block["start"],
                self.warmUp,
            )
            end = min(
                block["end"],
                self.maxInstance,
            )

            if start >= end:
                continue

            attackName = block["attack_name"]
            attackColor = self.attackColors.get(
                attackName,
                "#777777",
            )

            axis.axvspan(
                start,
                end,
                facecolor=attackColor,
                alpha=0.22,
                linewidth=0,
                zorder=1,
            )

            textLevel = 0.96 if index % 2 == 0 else 0.80

            axis.text(
                (start + end) / 2,
                textLevel,
                attackName,
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=6.0,
                fontweight="bold",
                color=attackColor,
                bbox={
                    "facecolor": "white",
                    "edgecolor": attackColor,
                    "linewidth": 0.30,
                    "alpha": 0.88,
                    "boxstyle": "round,pad=0.06",
                },
                clip_on=True,
                zorder=10,
            )

        for abbreviation, metric, deviation in (
            ("FP", "FP_avg", "FP_std"),
            ("FN", "FN_avg", "FN_std"),
        ):
            mean = series[metric].to_numpy(dtype=float)

            if deviation in series.columns:
                standardDeviation = (
                    series[deviation]
                    .fillna(0)
                    .to_numpy(dtype=float)
                )
            else:
                standardDeviation = np.zeros_like(mean)

            validMask = (
                evaluationMask
                & np.isfinite(x)
                & np.isfinite(mean)
                & np.isfinite(standardDeviation)
            )
            validX = x[validMask]
            validMean = mean[validMask]
            validStandardDeviation = standardDeviation[validMask]

            axis.fill_between(
                validX,
                np.maximum(
                    validMean - validStandardDeviation,
                    0,
                ),
                np.minimum(
                    validMean + validStandardDeviation,
                    self.yUpperLimit,
                ),
                color=self.errorColors[abbreviation],
                alpha=0.09,
                linewidth=0,
                zorder=3,
            )

            axis.plot(
                validX,
                validMean,
                color=self.errorColors[abbreviation],
                linewidth=0.7,
                solid_capstyle="round",
                zorder=6,
            )

        modelName = self.compactModelName(row)
        titleType = (
            "Classifier"
            if taskType == "classifier"
            else "Anomaly Detector"
        )
        title = (
            f"{titleType}\n"
            f"{configuration['name']} . "
            f"{configuration['block']}\n"
            f"({panelLetter}) {modelName}"
        )

        axis.set_title(
            title,
            pad=3.5,
            fontsize=7.8,
            fontweight="semibold",
            linespacing=1.08,
        )
        axis.set_xlim(*self.xLimits)
        axis.set_ylim(*self.plotYLimits)
        axis.xaxis.set_major_locator(
            FixedLocator(self.xTicks)
        )
        axis.xaxis.set_major_formatter(
            FuncFormatter(self.formatInstance)
        )
        axis.yaxis.set_major_locator(
            FixedLocator(self.yTicks)
        )
        axis.yaxis.set_major_formatter(
            FuncFormatter(self.formatError)
        )
        axis.tick_params(
            axis="both",
            which="major",
            pad=1.8,
            length=2.8,
            width=0.70,
        )
        axis.grid(
            True,
            linestyle=":",
            linewidth=0.55,
            alpha=0.38,
            zorder=0,
        )

        for spine in axis.spines.values():
            spine.set_linewidth(0.70)

    def createLegend(self):
        return [
            Line2D(
                [0],
                [0],
                color=self.errorColors["FP"],
                linewidth=1.0,
                label="FP",
            ),
            Line2D(
                [0],
                [0],
                color=self.errorColors["FN"],
                linewidth=1.0,
                label="FN",
            ),
            mpatches.Patch(
                facecolor="#BDBDBD",
                alpha=0.45,
                label="Warm-up (0–2k)",
            ),
            mpatches.Patch(
                facecolor=self.attackColors["DNS"],
                alpha=0.40,
                label="DNS",
            ),
            mpatches.Patch(
                facecolor=self.attackColors["SYN"],
                alpha=0.40,
                label="SYN",
            ),
            mpatches.Patch(
                facecolor=self.attackColors["LDAP"],
                alpha=0.40,
                label="LDAP",
            ),
        ]

    def createFigure(self):
        figure, axes = plt.subplots(
            2,
            4,
            figsize=(7.16, 4.00),
            sharex=True,
            sharey=True,
            facecolor="white",
        )

        classifierLetters = ["a", "b", "c", "d"]
        detectorLetters = ["e", "f", "g", "h"]

        for column, scenario in enumerate(self.scenarios):
            self.drawPanel(
                axes[0, column],
                scenario,
                "classifier",
                classifierLetters[column],
            )
            self.drawPanel(
                axes[1, column],
                scenario,
                "anomaly",
                detectorLetters[column],
            )

        for axis in axes[:, 1:].flat:
            axis.tick_params(
                axis="y",
                labelleft=False,
            )

        for axis in axes.flat:
            axis.tick_params(
                axis="x",
                labelbottom=True,
            )

        figure.supxlabel(
            "Stream instance",
            x=0.535,
            y=0.095,
            fontsize=8.5,
            fontweight="semibold",
        )
        figure.supylabel(
            "Errors per 100 instances",
            x=0.012,
            y=0.51,
            fontsize=8.5,
            fontweight="semibold",
        )
        figure.legend(
            handles=self.createLegend(),
            loc="lower center",
            bbox_to_anchor=(0.5, 0.012),
            ncol=6,
            frameon=False,
            columnspacing=1.25,
            handlelength=1.8,
            handletextpad=0.42,
        )
        figure.subplots_adjust(
            left=0.072,
            right=0.995,
            top=0.91,
            bottom=0.19,
            hspace=0.72,
            wspace=0.10,
        )

        return figure

    def saveFigure(self, figure):
        self.outputDirectory.mkdir(
            parents=True,
            exist_ok=True,
        )

        baseName = "BestModels_FP_FN_AllScenarios_2x4"
        pdfFile = self.outputDirectory / f"{baseName}.pdf"
        pngFile = self.outputDirectory / f"{baseName}.png"

        figure.savefig(
            pdfFile,
            format="pdf",
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor="white",
        )
        figure.savefig(
            pngFile,
            format="png",
            dpi=400,
            bbox_inches="tight",
            pad_inches=0.025,
            facecolor="white",
        )

        return pdfFile, pngFile

    def generate(self):
        """Run the complete workflow and return the saved file paths."""
        self.configureStyle()
        allCsvFiles = self.findResults()
        self.loadMetadata()
        self.selectModels(allCsvFiles)
        self.loadRuns()
        self.configureAxes()

        figure = self.createFigure()
        pdfFile, pngFile = self.saveFigure(figure)

        if self.showFigure:
            plt.show()

        plt.close(figure)

        print(f"Saved: {pdfFile}")
        print(f"Saved: {pngFile}")

        return {
            "pdf": pdfFile,
            "png": pngFile,
            "models": self.selectedModels.copy(),
        }
