"""
Extending lung dataset with other standard plane acquitions of other region of interest:
- Liver
- Heart (PLAX) 

The code create 5 fake ptient for liver a heart view to be added:
- dataset path: adding the folder with images and lable (null)
- 5CV: adding the name to splitting.json
"""

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Create the LUS dataset from OpenPOCUS data")
    parser.add_argument("--dataset_path", type=str, help="The path to the dataset")
    parser.add_argument("--dataset_ext", type=str, help="The path to the extended dataset")
                        
    args = parser.parse_args()

    main(args)
