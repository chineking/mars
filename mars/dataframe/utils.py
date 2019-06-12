# Copyright 1999-2018 Alibaba Group Holding Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import operator

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None
try:
    import cudf
except ImportError:
    cudf = None

from ..lib.mmh3 import hash as mmh_hash


def hash_index(index, size):
    def func(x, size):
        return mmh_hash(bytes(x)) % size

    f = functools.partial(func, size=size)
    grouped = sorted(index.groupby(index.map(f)).items(),
                     key=operator.itemgetter(0))
    return [g[1] for g in grouped]


def hash_dtypes(dtypes, size):
    hashed_indexes = hash_index(dtypes.index, size)
    return [dtypes[index] for index in hashed_indexes]


def convert_to_pandas_df(df):
    if pd is None:
        raise TypeError('Cannot convert to pandas DataFrame because pandas not installed')

    if isinstance(df, pd.DataFrame):
        return df
    if cudf and isinstance(df, cudf.DataFrame):
        return df.to_pandas()

    raise TypeError('Unknown dataframe type: {0}'.format(type(df)))


def convert_to_cudf_df(df):
    if cudf is None:
        raise TypeError('Cannot convert to cudf DataFrame because cudf not installed')

    if isinstance(df, cudf.DataFrame):
        return df
    if isinstance(df, pd.DataFrame):
        return cudf.DataFrame.from_pandas(df)

    raise TypeError('Unknown dataframe type: {0}'.format(type(df)))
