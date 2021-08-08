import kaggle as kg


def begin_download():
    '''This just starts the kaggle api and downloads a file list, and then the files.
    You should make sure you have already signed up to Kaggle however and gotten an api key or it won't work.
    The data is around 70gbs, so you will need to run this for a while, maybe watch a movie..but not online of course.
    This only downloads to the working folder,and not the data folder, user will still have to unzip as the kaggle app
    doesn't seem to accept path or unzip commands.
    '''

    competition= "g2net-gravitational-wave-detection"
    api = kg.KaggleApi()
    api.authenticate()
    #g2net-gravitational-wave-detection
    #print(api.competitions_data_list_files(competition))#uncomment if you want to see the files.
    api.competition_download_files(competition)


