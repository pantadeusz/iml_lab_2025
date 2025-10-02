import pandas as pd
import matplotlib.pyplot as plt
import sys

    
def run():
    args = get_args()
    filtered_data = prepare_df(args)
    plot_and_save(filtered_data, args['column'], args['min_bound'], args['max_bound'])
    print('Finished')
    
def get_args():
    file_path = sys.argv[1]
    column = sys.argv[2]
    min_bound = float(sys.argv[3])
    max_bound = float(sys.argv[4])
    
    return {'file_path': file_path,
            'column': column,
            'min_bound': min_bound,
            'max_bound': max_bound}

def prepare_df(kwargs):
    df = pd.read_csv(kwargs['file_path'])
    
    filtered_df = df[(df[kwargs['column']] > kwargs['min_bound']) & (df[kwargs['column']] < kwargs['max_bound'])]
    
    print(filtered_df[kwargs['column']].to_list())
    return filtered_df

def plot_and_save(df, column, min_bound, max_bound):
    plt.hist(df[column], bins=80, color='purple')
    plt.title(f'Number of certain Audio Loudness from {min_bound} to {max_bound} levels')
    plt.xlabel('Audio Loundess Value')
    plt.ylabel('Quantity')
    plt.savefig('histogram.png')
    plt.show()

if __name__ == '__main__':
    run()