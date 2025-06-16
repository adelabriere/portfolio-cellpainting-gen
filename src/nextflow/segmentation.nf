

// Default parameter input
params.path_pipeline = "/mnt/c/Users/alexi/Documents/dev/portfolio/imaging/segmentation/cellprofiler/segmentation_pipeline.cppipe"
params.path_csv = "/mnt/c/Users/alexi/Documents/dev/portfolio/imaging/segmentation/cellprofiler/segmentation_pipeline.cppipe"
params.size_batch = 20
params.path_script = "split_csv.py"

params.output_dir = "/mnt/c/Users/alexi/Documents/dev/portfolio/imaging/segmentation/test_results"


// splitString process
process SplitCSV {
    input:
    path input_csv
    val size_batch
    
    output:
    path 'chunk_*.csv'

    script:
    """
    python split_csv.py $input_csv $size_batch $PWD"/chunk"
    """
}

// convertToUpper process
process ApplyCellProfilerPipeline {
    container "cellpro:latest"
    publishDir "results/upper"
    tag "$y"

    input:
    path pipeline
    path metadata

    output:
    path 'measures/*.csv'
    path 'overview/*.jpeg'


    script:
    """
    cellprofiler -c -r -p $pipeline -o $PWD --data-file $metadata
    """
}

workflow {
    SplitCSV(params.path_csv, params.size_batch)
    ApplyCellProfilerPipeline(params.path_pipeline, "chunk_*.csv")
}