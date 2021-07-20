import kaggle as kg



#This just starts the kaggle api and downloads a file list, not the actual files.
def initialize(comp_name):
    api = kg.KaggleApi()
    api.authenticate()
    #g2net-gravitational-wave-detection
    print(api.competitions_data_list_files(comp_name))


