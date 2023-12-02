import argparse
from main_text import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--dataset", default="AG_NEWS")
    parser.add_argument("--num_train", type=int, default=100)
    parser.add_argument("--num_trials", type=int, default=5)
    parser.add_argument("--compressor", default="gzip")
    parser.add_argument("--para", action="store_true", default=False)
    parser.add_argument("--output_dir", default="text_exp_output")
    parser.add_argument("--test_idx_fn", default=None)
    parser.add_argument("--test_idx_start", type=int, default=None)
    parser.add_argument("--test_idx_end", type=int, default=None)
    parser.add_argument("--k", default=2, type=int)
    parser.add_argument("--class_num", default=5, type=int)
    args = parser.parse_args()
    # create output dir
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    train_idx_fn = os.path.join(
        args.output_dir,
        "{}_train_indicies_{}per_class".format(args.dataset, args.num_train),
    )
    # all dataset class number
    ds2classes = {
        "AG_NEWS": 4,
        "SogouNews": 5,
        "DBpedia": 14,
        "YahooAnswers": 10,
        "20News": 20,
        "Ohsumed": 23,
        "Ohsumed_single": 23,
        "R8": 8,
        "R52": 52,
        "kinnews": 14,
        "swahili": 6,
        "filipino": 5,
        "kirnews": 14,
        "custom": args.class_num,
    }
    # load dataset
    data_dir = os.path.join(args.data_dir, args.dataset)
    if args.dataset not in [
        "20News",
        "Ohsumed",
        "Ohsumed_single",
        "R8",
        "R52",
        "kinnews",
        "swahili",
        "filipino",
        "kirnews",
        "SogouNews",
        "custom",
    ]:
        dataset_pair = eval(args.dataset)(root=args.data_dir)
    else:
        if args.dataset == "20News":
            dataset_pair = load_20news()
        elif args.dataset == "Ohsumed":
            dataset_pair = load_ohsumed(args.data_dir)
        elif args.dataset == "Ohsumed_single":
            dataset_pair = load_ohsumed_single(args.data_dir)
        elif args.dataset == "R8" or args.dataset == "R52":
            dataset_pair = load_r8(args.data_dir)
        elif args.dataset == "kinnews":
            dataset_pair = load_kinnews_kirnews(
                dataset_name="kinnews_kirnews", data_split="kinnews_cleaned"
            )
        elif args.dataset == "kirnews":
            dataset_pair = load_kinnews_kirnews(
                dataset_name="kinnews_kirnews", data_split="kirnews_cleaned"
            )
        elif args.dataset == "swahili":
            dataset_pair = load_swahili()
        elif args.dataset == "filipino":
            dataset_pair = load_filipino(args.data_dir)
        else:
            dataset_pair = load_custom_dataset(args.data_dir)
    num_classes = ds2classes[args.dataset]
    
    train_pair, test_pair = dataset_pair[0], dataset_pair[1]
    test_data, test_labels = retrieve_text_labels(test_pair)

    accuracy_arr = np.array([])

    for i in range(args.num_trials):
        np.random.seed(i)
        # choose indices
        # pick certain number per class
        if args.test_idx_fn is not None or args.test_idx_start is not None:
            train_idx = np.load(train_idx_fn + ".npy")
            train_data, train_labels = read_torch_text_labels(
                dataset_pair[0], train_idx
            )
        else:
            train_data, train_labels = pick_n_sample_from_each_class_given_dataset(
                dataset_pair[0], args.num_train, train_idx_fn
            )
        trial_acc = non_neural_knn_exp(
            args.compressor,
            test_data,
            test_labels,
            train_data,
            train_labels,
            agg_by_concat_space,
            NCD,
            args.k,
            para=args.para
        )
        accuracy_arr = np.append(accuracy_arr, trial_acc)

    print(f"\nOVERALL ACCURACY: {accuracy_arr.mean():.3f} ± {accuracy_arr.std():.3f}")

