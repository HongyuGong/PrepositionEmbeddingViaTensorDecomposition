"""
error logs
"""

def countMissingHeadWord(vocab_fn, cands_fn, head_ind_fn):
    """
    count gold heads not in vocab
    count child not in vocab
    """
    f = open(vocab_fn, "r")
    vocab_list = f.readlines()
    vocab_list = [word.strip() for word in vocab_list]
    f.close()

    f = open(cands_fn, "r")
    cand_list = f.readlines()
    cand_list = [line.strip().split() for line in cand_list]
    f.close()

    f = open(head_ind_fn, "r")
    head_ind_list = f.readlines()
    head_ind_list = [int(ind.strip())-1 for ind in head_ind_list]
    f.close()

    head_list = []
    for inst_ind in range(len(cand_list)):
        head = cand_list[inst_ind][head_ind_list[inst_ind]]
        head_list.append(head)

    # check head not existing in vocab
    missing_count = 0
    for head in head_list:
        if (head not in vocab_list):
            missing_count += 1
    print "total head words:", len(head_list)
    print "missing head words:", missing_count

    

def countMissingChildrenWord(vocab_fn, child_fn):

    f = open(vocab_fn, "r")
    vocab_list = f.readlines()
    vocab_list = [word.strip() for word in vocab_list]
    f.close()
    
    f = open(child_fn, "r")
    child_list = [line.strip() for line in f.readlines()]
    f.close()

    # count missing child words
    missing_count= 0
    for child in child_list:
        if (child not in vocab_list):
            missing_count += 1
    print "total child words:", len(child_list)
    print "missing child words:", missing_count
    
    
    
    
if __name__=="__main__":
    data_folder = "../pp-attachment/data/pp-data-english/"
    train_prefix = "wsj.2-21.txt.dep.pp."
    test_prefix = "wsj.23.txt.dep.pp."
    vocab_fn = "dump/vocab.txt"

    # train data
    cands_fn = data_folder+train_prefix+"heads.words"
    head_ind_fn = data_folder+train_prefix+"labels"
    child_fn = data_folder+train_prefix+"children.words"
    countMissingHeadWord(vocab_fn, cands_fn, head_ind_fn)
    countMissingChildrenWord(vocab_fn, child_fn)

    # test data
    cands_fn = data_folder+test_prefix+"heads.words"
    head_ind_fn = data_folder+test_prefix+"labels"
    child_fn = data_folder+test_prefix+"children.words"
    countMissingHeadWord(vocab_fn, cands_fn, head_ind_fn)
    countMissingChildrenWord(vocab_fn, child_fn)
    
    

    
