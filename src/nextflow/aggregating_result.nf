// Pipeline ot aggregate the results and the different values

params.path_batches = "/mnt/c/Users/alexi/Documents/data/images/cellpainting/cpg0016-jump/data/results_target_docker"
params.path_output = "/mnt/c/Users/alexi/Documents/data/images/cellpainting/cpg0016-jump/data/results_target"

process AggregateResults {
    
    input:
    path folder

    output:
    path 'merged_aggregated.csv'

    script:
    """
    python $projectDir/bin/aggregate_by_well.py --input $folder --output .
    """
}

// List folder in the path_batches directory

process CopyFile {
    input:
    path filepath
    path newpath

    script:
    """
    cp $filepath $newpath
    """
}

workflow {
    froot = file(params.path_batches)
    filelists = Channel.of(froot.listFiles())

    // We check the existence of the file
    filtered_filelists = filelists.filter {
        def npath = froot / it.name / 'measures'
        npath.exists()
    }

    // Aggregate all files based on folder
    agg_files = AggregateResults(filtered_filelists)
    agg_file = agg_files.collectFile(name: "well_descriptors.csv",keepHeader:true, skip:1, storeDir: params.path_output)
    agg_file.view()
    // out_file = froot / "well_descriptors.csv"
    // CopyFile(agg_file, out_file)
}