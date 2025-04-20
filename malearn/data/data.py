from os.path import join

import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    StratifiedShuffleSplit,
    GroupShuffleSplit,
)
from sklearn.pipeline import make_pipeline
from sklearn.utils import compute_sample_weight
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer
from sklearn.utils.metadata_routing import _routing_enabled

from .preprocessing import *
from .utils import *

DEFAULT_NUMERICAL_TRANSFORMATION = "scale"
DEFAULT_CATEGORICAL_TRANSFORMATION = "onehot"


def get_data_handler(config, dataset_alias):
    if dataset_alias == "nhanes":
        data_handler = NHANESDataHandler(config)
    elif dataset_alias == "adni":
        data_handler = ADNIDataHandler(config)
    elif dataset_alias == "life":
        data_handler = LifeDataHandler(config)
    elif dataset_alias == "breast":
        data_handler = BreastDataHandler(config)
    elif dataset_alias == "fico":
        data_handler = FICODataHandler(config)
    elif dataset_alias == "pharyngitis":
        data_handler = PharyngitisDataHandler(config)
    else:
        raise ValueError(f"Invalid dataset alias: {dataset_alias}.")

    task_type = config["experiment"]["task_type"]
    allowed_task_types = (
        [data_handler.task_type]
        if isinstance(data_handler.task_type, str) else data_handler.task_type
    )
    if not task_type in allowed_task_types:
        raise ValueError(
            f"Invalid task type for dataset {dataset_alias}: {task_type}."
        )

    return data_handler


class DataHandler():
    def __init__(self, config):
        self.config = config

    @property
    def categorical_features(self):
        return self._categorical_features

    @property
    def numerical_features(self):
        # We treat ordinal features as numerical features.
        return self._numerical_features + self._ordinal_features

    @property
    def features(self):
        return self.categorical_features + self.numerical_features

    def _read(self, path):
        return pd.read_pickle(path)

    def _simple_impute(self, X):
        X[self._numerical_features] = X[self._numerical_features].fillna(
            X[self._numerical_features].mean(),
        )
        X[self._ordinal_features] = X[self._ordinal_features].fillna(
            X[self._ordinal_features].mode().squeeze(),
        )
        X[self._categorical_features] = X[self._categorical_features].fillna(
            X[self._categorical_features].mode().squeeze(),
        )
        return X

    def load_data(self):
        data_path = join(
            self.config["base_dir"], self.config["data_dir"], self.file_name
        )
        data = self._read(data_path)
        # Find feature columns with dtype `object` and encode them as integers,
        # as scikit-learn encoders do not accept strings mixed with NaN values.
        obj_cols = data[self.features].select_dtypes(include=["object"]).columns
        data[obj_cols] = data[obj_cols].astype("category").apply(lambda x: x.cat.codes)
        # Missing values now have the caterogical code `-1`.
        data[obj_cols] = data[obj_cols].replace(-1, np.nan)
        if (
            self.config["experiment"]["missing_features"] > 0
            and self.config["experiment"]["missing_rate"] > 0
        ):
            X = data[self.features].copy()
            X = self._simple_impute(X)
            seed = self.config["experiment"]["seed"]
            mechanism = self.config["experiment"]["missing_mechanism"]
            if mechanism == "mar":
                M = MAR_mask(
                    X.astype("float32").values,
                    p=self.config["experiment"]["missing_rate"],
                    # The proportion of features with no missing values.
                    p_obs=1-self.config["experiment"]["missing_features"],
                    seed=seed,
                )
            elif mechanism == "mnar":
                M = MNAR_mask_quantiles(
                    X.astype("float32").values,
                    p=self.config["experiment"]["missing_rate"],
                    q=0.25,
                    # The proportion of features with missing values.
                    p_params=self.config["experiment"]["missing_features"],
                    cut="both",
                    MCAR=False,
                    seed=seed,
                )
            else:
                raise ValueError(f"Invalid missingness mechanism: {mechanism}.")
            data[self.features] = data[self.features].mask(M, np.nan)
        sample_size = self.config["experiment"].get("sample_size", None)
        if sample_size is not None:
            data = data.sample(
                n=sample_size,
                random_state=self.config["experiment"]["seed"],
            )
        return data

    def split_data(self):
        data = self.load_data()
        X = data[self.features]
        y = data[self.target]
        if self.config["experiment"]["task_type"] == "classification":
            y = LabelEncoder().fit_transform(y)
        M = X.isna().astype(int)
        arrays = (X, y, M)
        X_train, X_test, y_train, y_test, M_train, M_test = train_test_split(
            *arrays,
            test_size=self.config["experiment"]["test_size"],
            random_state=self.config["experiment"]["seed"],
            shuffle=True,
            stratify=y,
        )
        return X_train, y_train, M_train, X_test, y_test, M_test

    def get_cv_splitter(self, n_splits=5, test_size=None, seed=None):
        return StratifiedShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=seed,
        )

    def _check_imputation_strategy(self, imputation_strategy):
        """Validate the imputation strategy.

        Parameters
        ----------
        imputation_strategy : str or dict
            The imputation strategy to validate.
            If a string, it must be one of:
                - "mean"
                - "zero"
                - "none"
            If a dictionary, it must have the following structure:
                - "numerical" : {"mean", "median", "zero", "mice"}
                - "categorical" : {"mode", "zero", "mice"}
        """
        if isinstance(imputation_strategy, str):
            if imputation_strategy not in {"mean", "zero", "none"}:
                raise ValueError(
                    f"Invalid string imputation strategy: {imputation_strategy}. "
                    "Valid options are 'mean', 'zero', or 'none'."
                )

        elif isinstance(imputation_strategy, dict):
            required_keys = {"numerical", "categorical"}
            if set(imputation_strategy) != required_keys:
                raise ValueError(
                    "Invalid dictionary keys in imputation_strategy. "
                    f"Expected keys are {required_keys}."
                )

            # Validate the value for "numerical"
            num_strategy = imputation_strategy["numerical"]
            if num_strategy not in {"mean", "median", "zero", "mice"}:
                raise ValueError(
                    f"Invalid numerical strategy: {num_strategy}. "
                    "Valid options are 'mean', 'median', 'zero' or 'mice'."
                )

            # Validate the value for "categorical"
            # TODO: Add "indicator" to the list of valid options.
            cat_strategy = imputation_strategy["categorical"]
            if cat_strategy not in {"mode", "zero", "mice"}:
                raise ValueError(
                    f"Invalid categorical strategy: {cat_strategy}. "
                    "Valid options are 'mode', 'zero', or 'mice'."
                )

        else:
            raise ValueError(
                "imputation_strategy must be either a string or a dictionary."
            )

    def _get_imputer(self, imputation_strategy, dtype=None, random_state=None):
        # The imputation strategy is already checked.
        if imputation_strategy == "mice":
            if dtype == "numerical":
                return IterativeImputer(
                    estimator=Ridge(random_state=random_state),
                    random_state=random_state,
                )
            elif dtype == "categorical":
                return IterativeImputer(
                    estimator=LogisticRegression(
                        penalty="l1", random_state=random_state,
                    ),
                    random_state=random_state,
                )
            else:
                raise ValueError(
                    f"Invalid dtype for MICE imputation: {dtype}. Valid "
                    "options are 'numerical' and 'categorical'."
                )
        if imputation_strategy == "mean":
            return  SimpleImputer(strategy="mean")
        if imputation_strategy == "median":
            return  SimpleImputer(strategy="median")
        if imputation_strategy == "mode":
            return SimpleImputer(strategy="most_frequent")
        if imputation_strategy == "zero":
            return SimpleImputer(strategy="constant", fill_value=0)
        if imputation_strategy == "none":
            return "passthrough"

    def _get_numerical_transformer(self, numerical_transformation):
        if numerical_transformation == "none":
            return "passthrough"

        if numerical_transformation == "discretize":
            encoder = MissingnessAwareKBinsDiscretizer()
        elif numerical_transformation == "scale":
            encoder = StandardScaler()
        else:
            raise ValueError(
                "Invalid transformation for numerical columns: "
                f"{numerical_transformation}. "
                "Valid options are 'discretize', 'scale' or 'none'."
            )

        return encoder.set_fit_request(sample_weight=False)

    def _get_categorical_transformer(self, categorical_transformation):
        if categorical_transformation == "onehot":
            return MissingnessAwareOneHotEncoder(
                drop="if_binary",
                handle_unknown="infrequent_if_exist",
                min_frequency=0.01,
                sparse_output=False,
            )

        elif categorical_transformation == "ordinal":
            return OrdinalEncoder()

        else:
            raise ValueError(
                "Invalid transformation for categorical columns: "
                f"{categorical_transformation}. "
                "Valid options are 'onehot' and 'ordinal'."
            )

    def get_preprocessor(self, estimator_alias):
        experiment_config = self.config["experiment"]
        estimator_config = self.config["estimators"][estimator_alias]

        numerical_transformation = estimator_config.get(
            "numerical_transformation", DEFAULT_NUMERICAL_TRANSFORMATION
        )
        categorical_transformation = estimator_config.get(
            "categorical_transformation", DEFAULT_CATEGORICAL_TRANSFORMATION
        )

        if estimator_config.get("keep_nan", False):
            imputation_strategy = "none"
        elif (
            experiment_config.get("numerical_imputation", None) is not None
            and experiment_config.get("categorical_imputation", None) is not None
        ):
            imputation_strategy = {
                "numerical": experiment_config["numerical_imputation"],
                "categorical": experiment_config["categorical_imputation"],
            }
        else:
            imputation_strategy = experiment_config["imputation"]

        self._check_imputation_strategy(imputation_strategy)

        seed = experiment_config["seed"]

        # Encode numerical columns before imputation to avoid bias.
        numerical_transformer = CustomColumnTransformer(
            transformers=[
                (
                    "numerical_transformer",
                    self._get_numerical_transformer(numerical_transformation),
                    self.numerical_features,
                )
            ],
            remainder="passthrough",
        )

        # Numerical columns will be followed by categorical columns after the
        # first transformation.
        n_numerical_features = len(self.numerical_features)
        n_features = len(self.features)
        numerical_columns = np.arange(n_numerical_features)
        categorical_columns = np.arange(n_numerical_features, n_features)

        # Impute missing values.
        if isinstance(imputation_strategy, str):
            imputer = self._get_imputer(imputation_strategy)
        else:
            numerical_imputer = (
                "numerical_imputer",
                self._get_imputer(
                    imputation_strategy["numerical"], "numerical", seed
                ),
                numerical_columns,
            )
            categorical_imputer = (
                "categorical_imputer",
                self._get_imputer(
                    imputation_strategy["categorical"], "categorical", seed
                ),
                categorical_columns,
            )
            imputer = CustomColumnTransformer(
                transformers=[numerical_imputer, categorical_imputer],
            )

        # Encode categorical columns.
        if numerical_transformation == "discretize":
            # All columns are categorical.
            categorical_transformer = self._get_categorical_transformer(
                categorical_transformation
            )
        else:
            categorical_transformer = CustomColumnTransformer(
            transformers=[
                (
                    "categorical_transformer",
                    self._get_categorical_transformer(categorical_transformation),
                    categorical_columns,
                )
            ],
            remainder="passthrough",
        )

        return make_pipeline(
            numerical_transformer, imputer, categorical_transformer
        )


class NHANESDataHandler(DataHandler):
    file_name = "NHANES_hypertension.pkl"
    task_type = "classification"
    _numerical_features = [
        "RIDAGEYR", "SMD415", "SMD415A", "BMXWT", "BMXHT", "BMXBMI",
        "BMXWAIST", "ALQ120Q", "DRXTTFAT", "DRXTSFAT", "LBXSAL", "LBXSGL",
        "LBXSCH", "LBXSUA", "LBXSKSI", "OCQ180", "OCD180", "DR1TTFAT",
        "DR1TSFAT", "INDFMMPI", "PAD645", "PAD660", "PAD675", "BMXHIP",
    ]
    _ordinal_features = []
    _categorical_features = [
        "RIAGENDR", "SMQ020", "SMD680", "PAD020", "PAD200", "PAD320", "DIQ010",
        "OHAROCDT", "OHAROCGP", "OHARNF", "MCQ160C", "MCQ080", "OCQ380",
        "DBD100", "SMD460", "SMDANY", "ALQ121", "MCQ366C",
    ]
    target = "HYPERT"

    def _read(self, data_path):
        data = pd.read_pickle(data_path)
        weights = compute_sample_weight(
            class_weight="balanced", y=data[self.target]
        )
        data = data.sample(
            n=10_000,
            axis=0,
            weights=weights,
            random_state=self.config["experiment"]["seed"]
        )
        return data


class ADNIDataHandler(DataHandler):
    file_name = "adni_clf.pkl"
    task_type = "classification"
    # `PIB`, `PTAU`, `TAU`, `ABETA`, `FLDSTRENG`, and `FSVERSION` are excluded
    # from the features.
    _numerical_features = [
        "AGE", "PTEDUCAT", "APOE4", "FDG", "AV45", "CDRSB", "ADAS11", "ADAS13",
        "ADASQ4", "MMSE", "LDELTOTAL", "TRABSCOR", "FAQ", "MOCA", "EcogPtMem",
        "EcogPtLang", "EcogPtVisspat", "EcogPtPlan", "EcogPtOrgan",
        "EcogPtDivatt", "EcogPtTotal", "EcogSPMem", "EcogSPLang",
        "EcogSPVisspat", "EcogSPPlan", "EcogSPOrgan", "EcogSPDivatt",
        "EcogSPTotal", "Ventricles", "Hippocampus", "WholeBrain", "Entorhinal",
        "Fusiform", "MidTemp", "ICV",
    ]
    _ordinal_features = []
    _categorical_features = ["PTGENDER", "PTETHCAT", "PTRACCAT", "PTMARRY"]
    target = "DX_change"

    def _read(self, data_path):
        data = pd.read_pickle(data_path)
        missing_values = ["Unknown"]
        data = data.replace(missing_values, np.nan)
        return data


class LifeDataHandler(DataHandler):
    file_name = "life.pkl"
    task_type = ["classification", "regression"]
    group = "Country"
    # `Year` and `Country` are excluded from the features.
    _numerical_features = [
        "Infant_deaths", "Under_five_deaths", "Adult_mortality",
        "Alcohol_consumption", "Hepatitis_B", "Measles", "BMI", "Polio",
        "Diphtheria", "Incidents_HIV", "GDP_per_capita", "Population_mln",
        "Thinness_ten_nineteen_years", "Thinness_five_nine_years", "Schooling",
    ]
    _ordinal_features = []
    _categorical_features = [
        "Economy_status_Developed", "Economy_status_Developing", "Region",
    ]
    target = "Life_expectancy"

    def _read(self, data_path):
        data = pd.read_pickle(data_path)
        data = data[data.Country.ne("missing")]
        country_to_region = {
            "Spain": "European Union",
            "Ukraine": "Rest of Europe",
            "missing": "missing",
            "Norway": "Rest of Europe",
            "Uruguay": "South America",
            "Yemen, Rep.": "Middle East",
            "Afghanistan": "Asia",
            "Oman": "Middle East",
            "Malaysia": "Asia",
            "Czechia": "European Union",
            "Chile": "South America",
            "Trinidad and Tobago": "Central America and Caribbean",
            "Serbia": "Rest of Europe",
            "Bahrain": "Middle East",
            "Kazakhstan": "Asia",
            "St. Lucia": "Central America and Caribbean",
            "Philippines": "Asia",
            "Pakistan": "Asia",
            "Denmark": "European Union",
            "Tunisia": "Africa",
            "Senegal": "Africa",
            "Brunei Darussalam": "Asia",
            "New Zealand": "Oceania",
            "Kyrgyz Republic": "Asia",
            "Mauritania": "Africa",
            "Seychelles": "Africa",
            "Bosnia and Herzegovina": "Rest of Europe",
            "Guatemala": "Central America and Caribbean",
            "Cyprus": "European Union",
            "Gambia, The": "Africa",
            "Austria": "European Union",
            "Indonesia": "Asia",
            "Honduras": "Central America and Caribbean",
            "Benin": "Africa",
            "Luxembourg": "European Union",
            "Croatia": "European Union",
            "Uganda": "Africa",
            "Canada": "North America",
            "Iran, Islamic Rep.": "Middle East",
            "Equatorial Guinea": "Africa",
            "Congo, Dem. Rep.": "Africa",
            "Slovenia": "European Union",
            "Botswana": "Africa",
            "Algeria": "Africa",
            "Netherlands": "European Union",
            "Montenegro": "Rest of Europe",
            "Cambodia": "Asia",
            "China": "Asia",
            "Russian Federation": "Rest of Europe",
            "Antigua and Barbuda": "Central America and Caribbean",
            "United Kingdom": "Rest of Europe",
            "Cameroon": "Africa",
            "Bulgaria": "European Union",
            "Rwanda": "Africa",
            "Togo": "Africa",
            "Burundi": "Africa",
            "St. Vincent and the Grenadines": "Central America and Caribbean",
            "Nepal": "Asia",
            "Paraguay": "South America",
            "Tonga": "Oceania",
            "Haiti": "Central America and Caribbean",
            "Mozambique": "Africa",
            "Djibouti": "Africa",
            "Latvia": "European Union",
            "Moldova": "Rest of Europe",
            "Portugal": "European Union",
            "Comoros": "Africa",
            "United Arab Emirates": "Middle East",
            "Brazil": "South America",
            "Samoa": "Oceania",
            "Germany": "European Union",
            "Romania": "European Union",
            "Kiribati": "Oceania",
            "Belarus": "Rest of Europe",
            "Turkiye": "Middle East",
            "Qatar": "Middle East",
            "Mauritius": "Africa",
            "Burkina Faso": "Africa",
            "Vietnam": "Asia",
            "Nicaragua": "Central America and Caribbean",
            "Saudi Arabia": "Middle East",
            "Guinea-Bissau": "Africa",
            "Madagascar": "Africa",
            "North Macedonia": "Rest of Europe",
            "Grenada": "Central America and Caribbean",
            "Zambia": "Africa",
            "Thailand": "Asia",
            "Namibia": "Africa",
            "Argentina": "South America",
            "Panama": "Central America and Caribbean",
            "Barbados": "Central America and Caribbean",
            "Egypt, Arab Rep.": "Africa",
            "Ghana": "Africa",
            "Cote d\"Ivoire": "Africa",
            "Lithuania": "European Union",
            "Myanmar": "Asia",
            "Suriname": "South America",
            "Belize": "Central America and Caribbean",
            "Papua New Guinea": "Oceania",
            "Italy": "European Union",
            "Poland": "European Union",
            "Lao PDR": "Asia",
            "Malta": "European Union",
            "Bolivia": "South America",
            "El Salvador": "Central America and Caribbean",
            "France": "European Union",
            "Estonia": "European Union",
            "Kenya": "Africa",
            "Micronesia, Fed. Sts.": "Oceania",
            "Belgium": "European Union",
            "Guinea": "Africa",
            "Morocco": "Africa",
            "Chad": "Africa",
            "Iceland": "Rest of Europe",
            "Australia": "Oceania",
            "Peru": "South America",
            "Ireland": "European Union",
            "Bangladesh": "Asia",
            "Sweden": "European Union",
            "Angola": "Africa",
            "Azerbaijan": "Asia",
            "Cabo Verde": "Africa",
            "Hungary": "European Union",
            "Georgia": "Rest of Europe",
            "Costa Rica": "Central America and Caribbean",
            "Guyana": "South America",
            "Singapore": "Asia",
            "Bahamas, The": "Central America and Caribbean",
            "Lesotho": "Africa",
            "Mexico": "North America",
            "Slovak Republic": "European Union",
            "Japan": "Asia",
            "South Africa": "Africa",
            "Syrian Arab Republic": "Middle East",
            "Somalia": "Africa",
            "Sao Tome and Principe": "Central America and Caribbean",
            "Fiji": "Oceania",
            "Greece": "European Union",
            "Gabon": "Africa",
            "Sri Lanka": "Asia",
            "Ethiopia": "Africa",
            "Zimbabwe": "Africa",
            "Israel": "Middle East",
            "Eritrea": "Africa",
            "Mozambique": "Africa",
            "Antigua and Barbuda": "Central America and Caribbean",
            "St. Lucia": "Central America and Caribbean",
        }
        data["Region"] = data.apply(
            lambda row: (
                country_to_region.get(row["Country"], "missing")
                if row["Region"] == "missing" else row["Region"]
            ),
            axis=1,
        )
        data["Region"] = data["Region"].replace("missing", np.nan)
        if self.config["experiment"]["task_type"] == "classification":
            data[self.target] = (
                data[self.target] > data[self.target].median()
            ).astype(int)
        return data

    def split_data(self):
        data = self.load_data()
        X = data[self.features]
        y = data[self.target]
        groups = data[self.group]

        if self.config["experiment"]["task_type"] == "classification":
            y = LabelEncoder().fit_transform(y)
        else:
            y = y.to_numpy()

        M = X.isna().astype(int)

        gss = GroupShuffleSplit(
            n_splits=1,
            test_size=self.config["experiment"]["test_size"],
            random_state=self.config["experiment"]["seed"],
        )
        train_index, test_index = next(gss.split(X, y, groups))

        X_train = X.iloc[train_index]
        y_train = y[train_index]
        M_train = M.iloc[train_index]

        X_test = X.iloc[test_index]
        y_test = y[test_index]
        M_test = M.iloc[test_index]

        return X_train, y_train, M_train, X_test, y_test, M_test

    def get_cv_splitter(self, n_splits=5, test_size=None, seed=None):
        splitter = GroupShuffleSplit(
            n_splits=n_splits,
            test_size=test_size,
            random_state=seed,
        )
        if _routing_enabled():
            return splitter.set_split_request(groups=True)
        return splitter


class BreastDataHandler(DataHandler):
    file_name = "breast_cancer.xlsx"
    task_type = "classification"
    _numerical_features = [
        "Fraction_Genome_Altered", "Invasive_Carcinoma_Diagnosis_Age",
        "Metastatic_Recurrence_Time", "Mutation_Count",
    ]
    _ordinal_features = [
        "ER_Status_of_the_Primary", "Metastatic_Disease_at_Last_Follow-up",
        "M_Stage", "N_Stage", "Overall_Patient_HER2_Status",
        "Overall_Patient_HR_Status", "Overall_Primary_Tumor_Grade",
        "PR_Status_of_the_Primary", "Stage_At_Diagnosis", "T_Stage",
    ]
    _categorical_features = ["Oncotree_Code", "Overall_Patient_Receptor_Status"]
    target = "Overall_Survival_Status"

    def _read(self, path):
        # Load data and remove duplicates.
        data = pd.read_excel(path).set_index("Patient ID")
        data = data[~data.index.duplicated(keep="first")]

        # Set missing values to `np.nan`.
        missing_values = ["Not Available", "unk", "Unk/ND", "UNk/ND", "Unknown"]
        data = data.replace(missing_values, np.nan)

        # Encode ordinal features.
        data = data.replace(
            {
                "ER Status of the Primary": {
                    "Positive": 1,
                    "Negative": 0,
                },
                "Metastatic Disease at Last Follow-up": {
                    "Yes": 1,
                    "No": 0,
                },
                "M Stage": {
                    "M1": 1,
                    "M0": 0,
                },
                "N Stage": {
                    "N0": 0, "N0(i+)": 0, "NX": 0,
                    "N1": 1, "N1a": 1, "N1b": 1, "N1c": 1, "N1mi": 1,
                    "N2": 2, "N2a": 2, "N2b": 2,
                    "N3": 3, "N3a": 3, "N3b": 3, "N3c": 3,
                },
                "Overall Patient HER2 Status": {
                    "Positive": 1,
                    "Negative": 0,
                },
                "Overall Patient HR Status": {
                    "Positive": 1,
                    "Negative": 0,
                },
                "Overall Primary Tumor Grade": {
                    "I  Low Grade (Well Differentiated)": 1,
                    "II  Intermediate Grade (Moderately Differentiated)": 2,
                    "III High Grade (Poorly Differentiated)": 3,
                },
                "PR Status of the Primary": {
                    "Positive": 1,
                    "Negative": 0,
                },
                "Stage At Diagnosis": {
                    "IA": 1, "IB": 2, "IIA": 3, "IIB": 4,
                    "IIIA": 5, "IIIB": 6, "IIIC": 7, "IV": 8,
                },
                "T Stage": {
                    "T0": 0, "Tis": 0, "TX": 0,
                    "T1": 1, "T1a": 1, "T1b": 1, "T1c": 1, "T1C": 1, "T1mi": 1,
                    "T2": 2,
                    "T3": 3,
                    "T4": 4, "T4a": 4, "T4b": 4, "T4c": 4, "T4d": 4,
                },
            }
        )

        # Remove spaces from column names.
        data.columns = data.columns.str.replace(" ", "_")

        # There is a single row per patient.
        data.reset_index(inplace=True, drop=True)

        return data


class FICODataHandler(DataHandler):
    file_name = "fico.csv"
    task_type = "classification"
    _numerical_features = [
        "ExternalRiskEstimate", "MSinceOldestTradeOpen",
        "MSinceMostRecentTradeOpen", "AverageMInFile", "NumSatisfactoryTrades",
        "NumTrades60Ever2DerogPubRec", "NumTrades90Ever2DerogPubRec",
        "PercentTradesNeverDelq", "MSinceMostRecentDelq",
        "MaxDelq2PublicRecLast12M", "MaxDelqEver", "NumTotalTrades",
        "NumTradesOpeninLast12M", "PercentInstallTrades",
        "MSinceMostRecentInqexcl7days", "NumInqLast6M", "NumInqLast6Mexcl7days",
        "NetFractionRevolvingBurden", "NetFractionInstallBurden",
        "NumRevolvingTradesWBalance", "NumInstallTradesWBalance",
        "NumBank2NatlTradesWHighUtilization", "PercentTradesWBalance",
    ]
    _ordinal_features = []
    _categorical_features = []
    target = "RiskPerformance"

    def _read(self, path):
        data = pd.read_csv(path)
        missing_values = [-7, -8, -9]
        data = data.replace(missing_values, np.nan)
        return data


class PharyngitisDataHandler(DataHandler):
    file_name = "pharyngitis.xls"
    task_type = "classification"
    _numerical_features = ["age_y", "temperature"]
    _ordinal_features = ["swollenadp"]
    _categorical_features = [
        "pain", "tender", "tonsillarswelling", "exudate", "sudden", "cough",
        "rhinorrhea", "conjunctivitis", "headache", "erythema", "petechiae",
        "abdopain", "diarrhea", "nauseavomit", "scarlet",
    ]
    target = "radt"

    def _read(self, path):
        data = pd.read_excel(path)
        # Number is a unique identifier.
        data = data.drop(columns=["number"])
        return data
