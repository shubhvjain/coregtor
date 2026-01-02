DATA_SOURCES = {
    "hippie_ppi": {
      "url":"https://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/hippie_current.txt",
      "name":"hippie_ppi.txt",
      "about":"The latest version of HIPPIE dataset. See https://cbdm-01.zdv.uni-mainz.de/~mschaefer/hippie/download.php. The columns indicate (1) UniProt identifier and (2) Entrez Gene identifier of the first protein partner, (3) UniProt identifier and (4) Entrez Gene identifier of the second protein partner, (5) score and (6) a description summarizing the origin of the evidence for the interaction. If one gene maps to several proteins each combination of proteins is listed in a separate line. "
      },
    "string_ppi":{
      "url": "https://stringdb-downloads.org/download/protein.links.full.v12.0/9606.protein.links.full.v12.0.txt.gz",
      "name" : "string_ppi.txt.gz",
      "about":" The v12 of STRING database. See  https://string-db.org/cgi/download. "
    },
    "string_protein":{
      "url": "https://stringdb-downloads.org/download/protein.info.v12.0/9606.protein.info.v12.0.txt.gz",
      "name" : "string_protein.txt.gz",
      "about":"This file contains list of STRING proteins incl. their display names and descriptions"
    },
     "biomart_gene_mapping": {
        "url": "http://www.ensembl.org/biomart/martservice?query=<?xml version='1.0' encoding='UTF-8'?><!DOCTYPE Query><Query virtualSchemaName='default' formatter='TSV' header='1' uniqueRows='1' datasetConfigVersion='0.6'><Dataset name='hsapiens_gene_ensembl' interface='default'><Attribute name='ensembl_gene_id'/><Attribute name='external_gene_name'/><Attribute name='entrezgene_id'/><Attribute name='uniprotswissprot'/><Attribute name='refseq_mrna'/><Attribute name='description'/></Dataset></Query>",
        "name": "biomart_gene_mapping.txt",
        "about": "BioMart gene mappings for human genes including Ensembl IDs, gene symbols (HGNC), Entrez Gene IDs, UniProt IDs, and RefSeq accessions. Downloaded from Ensembl BioMart API. See https://www.ensembl.org/biomart"
    },
    "collectri":{
      "url":"https://github.com/saezlab/CollecTRI",
      "name":"human_CollecTRI.csv",
      "about":"Signed transcription factor (TF) - target gene interactions"
    }
}