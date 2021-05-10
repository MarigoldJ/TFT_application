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

GenericDataFormatter = data_formatters.base.GenericDataFormatter
DataTypes = data_formatters.base.DataTypes
InputTypes = data_formatters.base.InputTypes

class UlsanFormatter(GenericDataFormatter):
    """
    info.
    """

    _column_definition = [
        ('id', DataTypes.CATEGORICAL, InputTypes.ID),
        ('date', DataTypes.CATEGORICAL, InputTypes.TIME),
        ('energy', DataTypes.REAL_VALUED, InputTypes.TARGET),
        ('month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('week_of_year', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('day_of_month', DataTypes.CATEGORICAL, InputTypes.KNOWN_INPUT),
        ('temperature', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('wind_speed', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('wind_direction', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT),
        ('humidity', DataTypes.REAL_VALUED, InputTypes.OBSERVED_INPUT),
        ('cloud', DataTypes.CATEGORICAL, InputTypes.OBSERVED_INPUT)
    ]

    def __init__(self):
        """Initialises formatter."""

        self.identifiers = None
        self._real_scalers = None
        self._cat_scalers = None
        self._target_scaler = None
        self._num_classes_per_cat_input = None

    def split_data(self, df):
        valid_boundary = pd.to_datetime('2019-12-01 00:00:00')
        test_boundary = pd.to_datetime('2020-07-03 00:00:00')

        index = df['date']
        train = df.loc[index < valid_boundary]
        valid = df.loc[(index >= valid_boundary) & (index < test_boundary)]
        test = df.loc[(index >= test_boundary) & (df.index <= '2019-06-28')]

        self.set_scalers(train)

        return (self.transform_inputs(data) for data in [train, valid, test])

    def set_scalers(self, df):
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
        data_target = df[target_column].values
        self._real_scalers = sklearn.preprocessing.StandardScaler().fit(data_real_input)
        self._target_scaler = sklearn.preprocessing.StandardScaler().fit(data_target)

        # Format categorical scalers
        categorical_inputs = utils.extract_cols_from_data_type(
            DataTypes.CATEGORICAL, column_definitions,
            {InputTypes.ID, InputTypes.TIME}
        )
        ### not done...

        # Set categorical scaler outputs
        ### not done...


    def transform_inputs(self, df):
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


    def format_predictions(self, predictions):
        pass

    # Default params
    def get_fixed_params(self):
        pass

    def get_default_model_params(self):
        pass

