import ir_datasets
import pyterrier as pt


def main():
    if not pt.started():
        pt.init()

    dataset_document = ir_datasets.load("msmarco-document-v2")
    dataset_passage = ir_datasets.load("msmarco-passage-v2")
    docstore_document = dataset_document.docs_store()
    docstore_passage = dataset_passage.docs_store()

    # get random document and passage to start creating the docstore
    docstore_document.get('msmarco_doc_05_295334611')
    docstore_passage.get('msmarco_passage_14_8526676')

    # create the index with deduplication
    dataset = pt.get_dataset('irds:msmarco-passage-v2/dedup')
    indexer = pt.IterDictIndexer('./indices/msmarco-passage-v2_dedup', meta={"docno": 28})
    indexer.index(dataset.get_corpus_iter(), fields=['text'])


if __name__ == '__main__':
    main()
