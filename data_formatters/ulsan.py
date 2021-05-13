# coding=utf-8

"""
1. csv 파일 합쳐야함. (script_download_data.py) -> csv파일 수정
 - 시간 맞추기
 - 뭐 맟추기

2. _column_definition 값 정하기 (DataType, InputType)
 - ID : 지점, 지점명 (cat, id)
 - energy : 전력량 (real, target)

 - temperature : 기온 (real, 측정치)
 - wind_speed : 풍속 (real, 측정치)
 - wind_direction : 풍향 (cat, 측정치)
 - humidity : 습도 (real, 측정치)
 - cloud : 전운량 (cat, 측정치)

 - date : 날짜 (date, time)
 - month : 월 (cat, known)
 - week_of_year : 주 (cat, known)
 - day_of_month : 일 (cat, known)

3. split_data 정하기
 - 기간으로 분류 => train : valid : test = 6 : 2 : 2
 - train, valid는 obs data에서 가져다 씀
 - test는 fcst data에서 가져다 씀

"""

import data_formatters.base
import libs.utils as utils
import sklearn.preprocessing
import pandas as pd

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes

class UlsanFormatter(GenericDataFormatter):
    """
    info.
    """

    _column_definition = [
        ('id', DataTypes.CATEGORICAL, InputTypes.ID),
        ('date', DataTypes.DATE, InputTypes.TIME),
        ('energy', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('days_from_start', DataTypes.REAL_VALUED, InputTypes.KNOWN_INPUT),
        ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        # ('week_of_year', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('day_of_month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('temperature', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('wind_speed', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('wind_direction', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('humidity', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('cloud', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('Region', DataTypes.CATEGORICAL, InputTypes.STATIC_INPUT)
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    def split_data(self, df):
        """Splits data frame into training-validation-test data frames.

        This also calibrates scaling object, and transforms data for each split.

        Args:
            df: Source data frame to split.
            valid_boundary: Starting year for validation data
            test_boundary: Starting year for test data

        Returns:
            Tuple of transformed (train, valid, test) data.
        """

        valid_boundary = pd.to_datetime('2019-12-01 00:00:00')
        test_boundary = pd.to_datetime('2020-07-03 00:00:00')

        index = df['date']
        index = pd.to_datetime(index)
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
        test = df.loc[(index >= test_boundary)]

        print(df)           # debug
        print(df.columns)   # debug
        print('train >', train.shape)   # debug
        print('valid >', valid.shape)   # debug
        print('test >', test.shape)     # debug

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
        """Calibrates scalers using the data supplied.

        Args:
            df: Data to use to calibrate scalers.
        """

        print('Setting scalers with training data...')

        column_definitions = self.get_column_definition()
        id_column = utils.get_single_col_by_input_type(InputTypes.ID, column_definitions)
        target_column = utils.get_single_col_by_input_type(InputTypes.TARGET, column_definitions)

        # Extract identifiers in case required
        self.identifiers = list(df[id_column].unique())

        # Format real scalers
        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME}
        )
        data_real_input = df[real_inputs].values
        data_target = df[[target_column]].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data_real_input)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(data_target)

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME}
        )
        categorical_scalers = {}
        num_classes = []
        for col in categorical_inputs:
            # Set all to str so that we don't have mixed integer/string columns
            srs = df[col].apply(str)
            categorical_scalers[col] = sklearn.preprocessing.LabelEncoder().fit(srs.values)
            num_classes.append(srs.nunique())

        # Set categorical scaler outputs
        self._cat_scalers = categorical_scalers
        self._num_classes_per_cat_input = num_classes


    def transform_inputs(self, df):
        """ Performs feature transformations.

        This includes both feature engineering, preprocessing and normalisation.
        Args:
            df: Data frame to transform.
        Returns:
            Transformed data frame.
        """

        output = df.copy()

        if self._real_scalers is None and self._cat_scalers is None:
            raise ValueError('Scalers have not been set!')

        column_definitions = self.get_column_definition()

        real_inputs = utils.extract_cols_from_data_type(
            DataTypes.REAL_VALUED, column_definitions,
            {InputTypes.ID, InputTypes.TIME}
        )
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME}
        )

        # Format real inputs
        output[real_inputs] = self._real_scalers.transform(df[real_inputs].values)

        # Format categorical inputs
        for col in categorical_inputs:
            string_df = df[col].apply(str)
            output[col] = self._cat_scalers[col].transform(string_df)

        return output


    def format_predictions(self, predictions):

        output = predictions.copy()
        column_names = predictions.columns

        for col in column_names:
            if col not in {'forecast_time', 'identifier'}:
                output[col] = self._target_scaler.inverse_transform(predictions[col])

        return output

    # Default params
    def get_fixed_params(self):

        fixed_params = {
            'total_time_steps': 8 * 24, # 수정 필요
            'num_encoder_steps': 7 * 24, # 수정 필요
            'num_epochs': 100,
            'early_stopping_patience': 5,
            'multiprocessing_workers': 5,
        }

        return fixed_params

    def get_default_model_params(self):

        # model_params = {
        #     'dropout_rate': 0.3,
        #     'hidden_layer_size': 160,
        #     'learning_rate': 0.01,
        #     'minibatch_size': 64,
        #     'max_gradient_norm': 0.01,
        #     'num_heads': 1,
        #     'stack_size': 1
        # }
        model_params = {
            'dropout_rate': 0.3,
            'hidden_layer_size': 160,
            'learning_rate': 0.01,
            'minibatch_size': 64,
            'max_gradient_norm': 0.01,
            'num_heads': 1,
            'stack_size': 1
        }

        return model_params

