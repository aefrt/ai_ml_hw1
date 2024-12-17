import pickle
import pandas as pd
import numpy as np

# Работа с текстовыми колонками:

reg_types = [
    r'^[.,\-~\+/\d]+\s*@\s*[,.\-~\+/\d]+\((Nm|nm|NM|kgm|KGM|)@ rpm\)$',
    r'^[.,\-~\+/\d]+\s*(Nm|nm|NM|kgm|KGM|)\s*(@|at)\s*[,.\-~\+/\d]+\s*(rpm|RPM|)$',
    r'^[.,\-~\+/\d]+\s*(Nm|nm|NM|kgm|KGM|)\s*/\s*[,.\-~\+/\d]+\s*(rpm|RPM|)$',
    r'^[.,\-~\+/\d]+\s*(Nm|nm|NM|kgm|KGM|)$',
    r'^[.,\-~\+/\d]+\s*(Nm|nm|NM|kgm|KGM|)\([.,\-~\+/\d]+(kgm|)\)\s*(@|at)\s*[,.\-~\+/\d]+\s*(rpm|RPM|)$'
]

def transform_regtype_0(series):
    cur_df = series.str.split('(')

    # measurements
    vals = pd.DataFrame(pd.DataFrame(np.array(cur_df.values.tolist())[:,0])[0].str.split('@').values.tolist())
    # measurements units
    units = pd.DataFrame(pd.DataFrame(np.array(cur_df.values.tolist())[:,1])[0].str.split('@').values.tolist())

    # measurements + units
    finl_df = (vals + units)
    finl_df[1] = finl_df[1].str.rstrip(')')  # final preprocessing
    
    return finl_df

reg_types_transformations = {
    reg_types[0]: transform_regtype_0,
    reg_types[1]: lambda series : series.str.replace('at', '@').str.split('@'),
    reg_types[2]: lambda series : series.str.split('/'), 
    reg_types[3]: lambda s : pd.DataFrame(s).assign(**{'1':  np.nan}),
    reg_types[4]: lambda s : s.str.split('@')
}

def preprocess_text(df):
    df['mileage'] = df.mileage.str.rstrip('kmpl').str.rstrip('km/kg').astype('float')
    df['max_power'] = df['max_power'].str.rstrip('bhp').replace(' ', np.nan).astype('float')
    df['engine'] = df.engine.str.rstrip('CC').astype('float')

    return df

def transform_torque(df, reg_type):
    cur_ser = df.loc[df.torque.str.lstrip().str.rstrip().str.match(reg_type).astype(bool).fillna(False)].torque.dropna()
    if cur_ser.shape[0] == 0:
        return
    return pd.DataFrame(
        data=reg_types_transformations[reg_type](cur_ser).values.tolist(),
        columns=['torque', 'max_torque_rpm'],
        index=cur_ser.index
    )

def force_rpm(s):
    cur_df = pd.DataFrame(s, columns=['torque'])
    cur_df['max_torque_rpm'] = np.nan
    return cur_df
    
def transform_torque_col(df):
    transformed_types = [transform_torque(df, reg_type) for reg_type in reg_types]
    null_rows = [force_rpm(df.torque.loc[df.torque.isna()])]
    return pd.concat(transformed_types + null_rows).sort_index()

def transform_torque_final(df):
    # Неструктурированный текст теперь в двух колонках
    res_df = transform_torque_col(
        df
    ).apply(lambda s : s.str.replace(r'\s*(Nm|nm|NM|kgm|KGM|rpm|RPM|)\s*', '', regex=True), axis=1)

    # Исправляем проблемы с ними:
    res_df.torque = res_df.torque.str.replace(r'\(\S*\)', '', regex=True).astype('str')
    res_problem1 = res_df.max_torque_rpm.loc[res_df.max_torque_rpm.str.match(
        r'[\d,]+\+/\-[\d,]+'
    ).astype(bool).fillna(False)].dropna()
    index_problem1 = res_problem1.index
    res_problem1 = pd.DataFrame(
        data=res_problem1.str.replace(',', '', regex=False).str.split('+/-', regex=False).values.tolist(),
        index=res_problem1.index
    ).astype('float64').sum(axis=1).astype('str')
    res_df.loc[index_problem1, 'max_torque_rpm'] = res_problem1.values

    res_problem2 = res_df.max_torque_rpm.loc[res_df.max_torque_rpm.str.match(
        r'[\d,]+\-[\d,]+'
    ).astype(bool).fillna(False)].dropna()
    index_problem2 = res_problem2.index
    res_problem2 = pd.DataFrame(
        data=res_problem2.str.replace(',', '', regex=False).str.split('-', regex=False).values.tolist(),
        index=res_problem2.index
    ).astype('float64').sum(axis=1).astype('str')
    res_df.loc[index_problem2, 'max_torque_rpm'] = res_problem2.values

    res_problem2 = res_df.max_torque_rpm.loc[res_df.max_torque_rpm.str.match(
        r'[\d,]+\~[\d,]+'
    ).astype(bool).fillna(False)].dropna()
    index_problem2 = res_problem2.index
    res_problem2 = pd.DataFrame(
        data=res_problem2.str.replace(',', '', regex=False).str.split('~', regex=False).values.tolist(),
        index=res_problem2.index
    ).astype('float64').sum(axis=1).astype('str')
    res_df.loc[index_problem2, 'max_torque_rpm'] = res_problem2.values

    # Осталось решить проблему с единицами измерения. Предполагаем что nM стандарт.
    res_df = res_df.apply(lambda x : x.str.replace(',', '', regex=False), axis=1).astype('float64')
    kgm_mask = df.torque.str.lower().str.contains('kgm').astype(bool).fillna(False)
    kgm_part = res_df.loc[kgm_mask, 'torque'].dropna()
    res_df.loc[kgm_part.index, 'torque'] *= 9.81  # nm = kgm * 9.81

    return pd.concat([df.drop(columns='torque'), res_df], axis=1)

def fillna_medians(df):
    numeric_cols = df.select_dtypes(np.number).columns
    medians = pd.read_pickle('medians.pkl')
    df.loc[:, numeric_cols] = df.loc[:, numeric_cols].fillna(medians)
    return df

def float_to_int(df):
    df['seats'] = df['seats'].astype('int')
    df['engine'] = df['engine'].astype('int')
    return df

def scale_numeric(df):
    numeric_cols = df.select_dtypes(np.number).columns
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    df.loc[:, numeric_cols] = scaler.transform(df.loc[:, numeric_cols])
    return df

def preprocess_name(df):
    df['name'] = df.name.str.split().str.slice(0, 2).str.join(' ')
    return df

def encode_categoric(df):
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    categoric_cols = df.select_dtypes('object').columns.tolist()
    
    df = pd.concat(
        [
            df.drop(columns=categoric_cols), 
            pd.DataFrame(
                data=encoder.transform(df[categoric_cols]), 
                columns=encoder.get_feature_names_out()
            )
        ], 
        axis=1
    )

    return df

def preprocess_data(df):
    df = preprocess_text(df)
    df = transform_torque_final(df)
    df = fillna_medians(df)
    df = float_to_int(df)
    df = scale_numeric(df)
    df = preprocess_name(df)
    df = encode_categoric(df)
    return df