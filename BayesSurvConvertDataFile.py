import pandas as pd
import datetime

def dateToRelativeMonthFloat(DF, dat_col):
    """
    +==========================================================================+
    units: months
    +==========================================================================+
    """
    # convert to datetime
    dates = pd.to_datetime(DF[dat_col])
    oldest = dates.min()
    dates.fillna(oldest, inplace=True)
    DF['StartTime'] = (dates - oldest).dt.days/30 #months

    #data = DF.iloc[:,[0,1,4,2]].to_numpy()

def relativeMonthFloatToDate(DF, dat_col, start_col):
    """
    +==========================================================================+
    units: months
    +==========================================================================+
    """
    dates = ["" for i in range(len(DF))]
    for i in range(len(DF)):
        dates[i] = (DF[start_col].iloc[i] + \
           datetime.timedelta(days=DF[dat_col].iloc[i]*30)).strftime('%m/%d/%Y')
    DF['EndTime [date]'] = dates


if __name__ == "__main__":

        # PembroMono
        file = "pembromono_OS_trial_rw_inclstartdate.csv"
        dataDF = pd.read_csv(file)

        print(dataDF.head())
        dataDF['V4'].fillna('14/07/2017', inplace=True) # start time, oldest item
        dataDF['VX'] = pd.to_datetime(dataDF['V4'])

        relativeMonthFloatToDate(dataDF, 'V1', 'VX')
        dateToRelativeMonthFloat(dataDF, 'V4')

        dataDF = dataDF[['V1', 'V2', 'V3', 'V4', 'EndTime [date]', 'StartTime']]
        dataDF.rename(columns={'V1' : 'EndTime', 'V2' : 'Censored', 'V3' : 'RW', 
                        'V4' : 'StartTime [date]'}, inplace=True)
        dataDF['TR'] = dataDF['RW']
        dataDF['RW'] = 1 - dataDF['TR']
        dataDF = dataDF[['EndTime', 'StartTime', 'Censored', 'RW', 'TR', 
                'EndTime [date]', 'StartTime [date]']]
        dataDF.to_csv('PBMonoOS_TRRW_start.csv', index=False)

        # PembroChemo
        file = "immunochemo_OS_trial_rw_inclstartdate.csv"
        dataDF = pd.read_csv(file)

        print(dataDF.head())
        dataDF['V4'].fillna('02/12/2018', inplace=True) # start time, oldest item
        dataDF['VX'] = pd.to_datetime(dataDF['V4'])

        relativeMonthFloatToDate(dataDF, 'V1', 'VX')
        dateToRelativeMonthFloat(dataDF, 'V4')

        dataDF = dataDF[['V1', 'V2', 'V3', 'V4', 'EndTime [date]', 'StartTime']]
        dataDF.rename(columns={'V1' : 'EndTime', 'V2' : 'Censored', 'V3' : 'RW', 
                        'V4' : 'StartTime [date]'}, inplace=True)
        dataDF['TR'] = dataDF['RW']
        dataDF['RW'] = 1 - dataDF['TR']
        dataDF = dataDF[['EndTime', 'StartTime', 'Censored', 'RW', 'TR', 
                'EndTime [date]', 'StartTime [date]']]
        dataDF.to_csv('PBChemoOS_TRRW_start.csv', index=False)