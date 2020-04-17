# Machine Reading of Historical Events

This is the repository containing the code for the models in the ACL 2020 paper "Machine Reading of Historical Events".

## Using the Datasets

The datasets are stored in the `data` folder. We encourage re-using the same splits (in the `train`, `validation` and `test` folders, respectively).
Each folder contains a `wotd` file corresponding to the Wikipedia On This Day (WOTD) dataset,
and a `otd2` file corresponding to the [On This Day 2 (OTD2) dataset](OTD2.md).
The data for the On This Day (OTD) dataset is available upon request.

The datasets are stored as pandas dataframes in pickle files. Loading them should be as simple as:

```
In [1]: import pandas as pd

In [2]: data = pd.read_pickle("data/validation/wotd.pkl")

In [3]: data.head()
Out[3]:
     YY                                              Event                                        Information
0  1949  Cold War: The western occupying powers approve...  [However, hundreds of thousands of East German...
1  1887  Buffalo Bill Cody's Wild West Show opens in Lo...  [In December 1872, Cody traveled to Chicago to...
2  1993  An election takes place in Nigeria which is la...  [=\nBabangida, then a lieutenant with the 1st ...
3  1601  Long War: Austria captures Transylvania in the...  [The Long Turkish War or Thirteen Years' War w...
4  1925  The Government of Turkey expels Patriarch Cons...                                                 []
```

The key key columns are:
- `YY`: The year of the event.
- `Event`: The event description.
- `Information`: An array, possibly empty, of Contextual Information (CI) sentences extracted for that event.

## Models

There are two models, you can find them under the appropriate `lstm` and `boe` folders.
Instructions on how to setup each of the models and reproduce experimental results are located there.

## Citation
```
@inproceedings{honovich2020historical,
    title = "Machine Reading of Historical Events",
    author = "Honovich, Or  and
      Torroba Hennigen, Lucas  and
      Abend, Omri  and
      Cohen, Shay B.",
    booktitle = "Proceedings of the 58th Conference of the Association for Computational Linguistics",
    year = "2020"
}

```
