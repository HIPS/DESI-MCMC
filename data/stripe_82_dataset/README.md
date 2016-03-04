## Stripe 82 Validation Set

This dataset includes catalog parameters for stars and galaxies in a small
region of stripe 82.  One set of parameters result from the PHOTO system
fit to co-added images (~50 images of the same piece of sky), and are
considered "ground truth" in a sense.

The second set of parameters result from the PHOTO system fit to a single
exposure (or maybe two?), and are considered a baseline.

The two files below include the coadd and primary (non-coadd) datasets,
and the included `load_stripe82_square.py` file to spatially align the two
datasets.

### Files:

* `square_106_4.fit` includes a catalog fit to the co-added data.  This file
resulted from the casjobs query:

```
SELECT *
INTO MyDB.square_106_4
FROM photoobj
WHERE ra BETWEEN 37.5 AND 38.5 
  AND dec > 0 AND dec < 1
  AND run = 106
  AND camcol = 4
```

* `square_6425_4.fit` includes a non co-add fit to parameters.  This file
resulted from the casjobs query:

```
SELECT *
INTO MyDB.square_6425_4
FROM photoobj
WHERE ra BETWEEN 37.5 AND 38.5 
  AND dec > 0 AND dec < 1
  AND run = 6425
  AND camcol = 4
```

* `square_4263_4.fit` includes another non-coadd fit to the same area of sky,
but uses a DR8+ run (4263).

```
SELECT *
INTO MyDB.square_4263_4
FROM Stripe82.photoobj
WHERE ra BETWEEN 37.5 AND 38.5 
  AND dec > 0 AND dec < 1
  AND run = 4263
  AND camcol = 4
```

