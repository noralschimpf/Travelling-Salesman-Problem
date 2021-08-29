import csv, os, numpy as np

def load_tsp(path):
    file = open(path, newline='')
    csvrd = csv.reader(file, delimiter=' ')
    nda_data = np.full((1,3), np.nan)
    # Loading all of TSP file into memory
    data = {}
    for row in csvrd:
        # Add metadata by name/tag
        if ':' in row[0]:
            data[row[0][:-1]] = ' '.join(row[1:])
        elif row[0] == 'NODE_COORD_SECTION': continue
        # Generate Numpy array of all cities in TSP file
        elif len(row) == 3:
            nda_row = np.array([int(row[0]), float(row[1]), float(row[2])]).reshape(1,3)
            if np.isnan(nda_data[0,0]): nda_data = nda_row
            else: nda_data = np.concatenate((nda_data, nda_row))
    file.close()
    data['data'] = nda_data
    return data
