from transformers import AutoModel, AutoTokenizer

FOUNDATIONMODELS = {
    "DNABERT": "zhihan1996/DNA_bert_6",                                                 # For DNA sequences, pretrained on k-mer representations
    "NucleotideTransformer": "InstaDeepAI/nucleotide-transformer-2.5b-multi-species",   # For DNA sequences, InstaDeepAI general-purpose DNA model
    #"GenSLMs": "some_model_identifier",                                                 # For DNA sequences, trained on microbial genomes
    "HyenaDNA": "some_model_identifier",                                                # For DNA sequences, ultra-long sequences
    "RNATransformer": "some_model_identifier",                                          # For RNA sequences
}

def fm_sequence_to_embedding(sequence, model_name=FOUNDATIONMODELS["NucleotideTransformer"]):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    inputs = tokenizer(
        sequence, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    )

    fm_embedding = model(**inputs).last_hidden_state.mean(dim=1)

    return fm_embedding.tolist()[0]
