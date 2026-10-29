# Co-Regulation Predictor 

Transcription of a gene is a highly regulated process with multiple Transcription Factors (TF) working in coordination with each other.  `CoRegTor` (which stands for CO-REGulation predicTOR) is a tool to predict potential coregulators of a specific target gene using gene expression data. It can also be used to find frequently occurring co-regulators within the gene expression data. 

## Installation 

```
pip install coregtor
```

## Documentation 

The documentation is available [here](https://shubhvjain.github.io/coregtor/)


[Tutorial to use the python package]()

[Tutorial to use the bulk pipeline]()

## Usage

### Data required
The 2 main inputs required are:
- The gene expression data where rows are samples and columns are genes. 
- A list transcription factors. Their names must match with the names of the genes name columns in the gene expression data. For humans, TF list is available [here](https://resources.aertslab.org/cistarget/tf_lists/). Additionally, you can also include the genes associated with the GO term [transcription coregulator activity](https://www.ebi.ac.uk/QuickGO/term/GO:0003712). Let 




Please see the [API doc](https://shubhvjain.github.io/coregtor/api.html#coregtor.RunCoRegTor) for more details on the parameters. 



