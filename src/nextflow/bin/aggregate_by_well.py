import pandas as pd
import os
import argparse
import functools

REMOVED_COLUMNS = ["AreaShape","Children","ObjectNumber"]
UNIQUE_COLUMNS=["FileName","PathName","ImageNumber"]

def is_contained(x,namesets):
    return any([(nn in x) for nn in namesets])

def aggregate_metrics(metrics,column,add_unique=True):
    cnames = metrics.columns
    if column not in cnames:
        raise KeyError(f"Column'{column}' not found.")
    # Removing positional measurement
    cnames = [x for x in cnames if not is_contained(x,REMOVED_COLUMNS)]
    u_cnames = [x for x in cnames if is_contained(x,UNIQUE_COLUMNS)]
    o_cnames = [x for x in cnames if x not in u_cnames]

    if column not in u_cnames:
        u_cnames = [column] + u_cnames

    if column not in o_cnames:
        o_cnames = [column] + o_cnames
    
    # Compute means
    metrics_o = metrics[o_cnames]
    agg_metrics = metrics_o.groupby(column).mean()

    if add_unique:
        # Compute unique values
        metrics_u = metrics[u_cnames]
        agg_unique = metrics_u.groupby(column).first()
        agg_metrics = agg_metrics.merge(agg_unique,left_index=True,right_index=True)
    return agg_metrics

def make_output_name(key,output_folder):
    return os.path.join(output_folder,key+"_aggregated.csv")

def aggregate_results_by_well(path_folder,output_dir):
    if not os.path.isdir(path_folder):
        raise ValueError("The specified folder {} does not exist".format(path_folder))


    # Getting the path of the different files
    path_measures = os.path.join(path_folder,"measures")
    path_metadata = [x for x in os.listdir(path_folder) if x.startswith("metadata")][0]
    path_metadata = os.path.join(path_folder,path_metadata)

    # Creating the output files if necessary
    # path_output = os.path.join(path_folder,"agg_results")
    # if not os.path.exists(path_output):
    #     os.makedirs(path_output)

    #Parsing and augemnting the metadata
    metas = pd.read_csv(path_metadata)
    metas["Metadata_WellPlate"] = metas.Metadata_WellPlateFrame.str[:-4]

    # We now generate all the measure file path
    measured_files = {
        "Cells":os.path.join(path_measures,"Cells.csv"),
        "Cytoplasm":os.path.join(path_measures,"Cytoplasm.csv"),
        "Nuclei":os.path.join(path_measures,"Nuclei.csv")        
    }

    to_concatenate = []
    
    for key,path in measured_files.items():
        metrics = pd.read_csv(path)
        min_image = metrics.ImageNumber.min()
        metrics.ImageNumber = metrics.ImageNumber-min_image

        # DEBUG TO REMOVE FATER RERUN
        metrics = metrics[metrics.ImageNumber<metas.shape[0]]

        # Addingunique well number
        metrics["WellPlate"] = metas.Metadata_WellPlate.loc[metrics.ImageNumber].reset_index(drop=True)

        agg_metrics = aggregate_metrics(metrics,column="WellPlate",add_unique=True)

        # Profiles to aggregate in a single file
        to_agg_metrics = agg_metrics.copy()
        
        # Adding the key label asd variable prefix
        new_names = {cname:("{}_{}".format(key,cname)) for cname in to_agg_metrics.columns}

        # We relabel the columns
        to_agg_metrics.rename(columns=new_names,inplace=True)
        to_concatenate.append(to_agg_metrics)

        path_output_file = os.path.join(make_output_name(key.lower(),output_dir))

        #_ = agg_metrics.to_csv(path_output_file)
    
    # We merge the table based on index
    merged_table = functools.reduce(lambda x,y : x.merge(y,left_index=True,right_index=True,how="outer"), to_concatenate)
    path_output_merged = make_output_name("merged",output_dir)
    merged_table.to_csv(path_output_merged)

    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Aggregate metrics by well')
    parser.add_argument('--input', type=str, help='Path to the input folder', default="/mnt/c//Users/alexi/Documents/data/images/cellpainting/cpg0016-jump/data/results_target_docker/batch8")
    parser.add_argument('--output', type=str, help='Path to the output folder')
    args = parser.parse_args()
    
    print("Aggregating results")
    aggregate_results_by_well(args.input,args.output)


