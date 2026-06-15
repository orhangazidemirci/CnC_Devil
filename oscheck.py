import os
cache = f'waterbirds_devil_split_5_seed42.pkl'
print(f'Cache exists: {os.path.exists(cache)}')
print(f'Cache path: {os.path.abspath(cache)}')

cache_path = 'waterbirds_devil_split_5_seed42.pkl'
if os.path.exists(cache_path):
    os.remove(cache_path)
    print(f'Deleted cache: {cache_path}')