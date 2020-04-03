# This script is meant to extend the odir annotations file's diagnostic code
# fields. The file contains separate keyword annotation for each
# side, but the code is given for both eyes together so we don't
# know if i.e 'g=1' refers to the left or right.
# I scanned the file semi manually and read the help file in order
# to decypher the annotation code. In addition to the encoded
# diagnostic keywords, there are exceptions of keywords that
# indicate the image itself is either not a retina but a whole eye
# image ('anterior' something...) or that there are two images of
# the same side (the wrong side has 'no fundus' keyword). This
# script encodes these special cases, for each eye separately,
# in their own binary fields.
# The output is a csv file and is printed to stdout for easy piping,
# unless the --output argument is used.
#####
# Find All Diagnostic Keywords and Encode Them with:
#####
# N,D,G,C,A,H,M,O
# N: normal
# D: ((non) proliferative) nonproliferative retinopathy
# G: glaucoma
# C: catarct
# A: age related macular degeneration
# H: hypertensive retinopathy
# M: myopia
# O: other diagnosys except 'anterior segment image' and 'no fonndus image'
# special keywords: 'anterior segment image',  'no fonndus image'


from __future__ import print_function, division
import os
import pandas as pd
from skimage import io, transform
import numpy as np
import argparse
import sys


# Command: python3 Preprocessing/decode_diagnostics_keywords.py /home/henrik/PycharmProjects/Project\ A\ -\ VAE\
# Retina/odir/ODIR-5K_Training_Annotations\(Updated\)_V2.xlsx --out /home/henrik/PycharmProjects/Project\ A\ -\ VAE\
# Retina/odir/decoded.csv

info_text = """Reads the odir
annotation file and asigns diagnostic codes for each side according
to the diagnostic keywords. So in addition to field 'N', 'D' etc.
The script adds fields 'LN','LD','RN','RD' etc.
In additional there is a field 'L-ant', 'R-ant' where a non-zero
indicates there is the special 'anterio segment image' keyword,
and 'L-no', 'R-no' which indicates the keyword 'no fundus image'.
Diagnostic keyword code:
N: normal
D: ((non) proliferative) nonproliferative retinopathy
G: glaucoma
C: catarct
A: age related macular degeneration
H: hypertensive retinopathy
M: myopia
O: other diagnosys except 'anterior segment image' and 'no fonndus image'
special keywords: 'anterior segment image',  'no fonndus image'
"""


def decode_d_k(xslxx_file, output_file="./data/odir/odir_train_lr_annotations.csv"):

    df = pd.read_excel(xsl_file)

    # get all the unique diagnostics as a list
    l = df["Left-Diagnostic Keywords"].tolist()
    l = np.unique(l).tolist()
    l = ",".join(l)
    l = l.split(",")
    l = np.unique(l).tolist()
    s = ",".join(l)
    s.replace(",", "")
    np.unique(l)
    s.split(",")
    x = l[-1]
    c = x[12]  # some weird char that looks like ', '
    s = s.replace(c, ",")
    l = s.split(",")
    l = np.unique(l).tolist()  # now l realy contains the unique

    # add separate left and right diagnotics columns instead of the
    # joined one:
    df["LN"] = np.zeros_like(df["N"])
    df["LD"] = np.zeros_like(df["D"])
    df["LG"] = np.zeros_like(df["G"])
    df["LC"] = np.zeros_like(df["C"])
    df["LA"] = np.zeros_like(df["A"])
    df["LH"] = np.zeros_like(df["H"])
    df["LM"] = np.zeros_like(df["M"])
    df["LO"] = np.zeros_like(df["O"])
    df["RN"] = np.zeros_like(df["N"])
    df["RD"] = np.zeros_like(df["D"])
    df["RG"] = np.zeros_like(df["G"])
    df["RC"] = np.zeros_like(df["C"])
    df["RA"] = np.zeros_like(df["A"])
    df["RH"] = np.zeros_like(df["H"])
    df["RM"] = np.zeros_like(df["M"])
    df["RO"] = np.zeros_like(df["O"])
    df["L-ant"] = np.zeros_like(df["O"])
    df["L-no"] = np.zeros_like(df["O"])
    df["R-ant"] = np.zeros_like(df["O"])
    df["R-no"] = np.zeros_like(df["O"])

    ### Find All Diagnostic Keywords and Encode Them with:
    feature = {
        "N": "normal fundus",
        "D": "proliferative retinopathy",
        "G": "glaucoma",
        "C": "catarct",
        "A": "age related macular degeneration",
        "H": "hypertensive retinopathy",
        "M": "myopia",
        "ant": "anterior segment",
        "no": "no fundus image",
    }

    # a function to search pattern in text
    f = lambda pattern: lambda text: (pattern in text)

    np.vectorize(f("normal"))(df["Left-Diagnostic Keywords"])

    # find features (except 'O') in Left, then Right Eye:
    for key, val in feature.items():
        testl = np.vectorize(f(val))(df["Left-Diagnostic Keywords"])
        testr = np.vectorize(f(val))(df["Right-Diagnostic Keywords"])
        if key == "no":
            df.loc[testl, "L-no"] = 1  # special case 'no fundus'
            df.loc[testr, "R-no"] = 1  # special case 'no fundus'
        elif key == "ant":
            df.loc[testl, "L-ant"] = 1  # special case 'ant'
            df.loc[testr, "R-ant"] = 1  # special case 'no fundus'
        else:
            df.loc[testl, "L" + key] = 1
            df.loc[testr, "R" + key] = 1

    # remove feature keywors off the list of diagnostics
    # so only 'O' Diagnostics remain:
    olist = l.copy()
    for w in l:
        for key, val in feature.items():
            if val in w:
                olist.remove(w)

    olist.remove("lens dust")
    olist.remove("optic disk photographically invisible")
    olist.remove("low image quality")
    olist.remove("image offset")

    # Now find the 'O' (=all other) diagnostics:
    for val in olist:
        testl = np.vectorize(f(val))(df["Left-Diagnostic Keywords"])
        testr = np.vectorize(f(val))(df["Right-Diagnostic Keywords"])
        df.loc[testl, "LO"] = 1
        df.loc[testr, "RO"] = 1

    # Making Left and Right each apear in separate row
    cols = df.columns.tolist()
    newcols = [
        "ID",
        "Side",
        "Patient Age",
        "Patient Sex",
        "Fundus Image",
        "Diagnostic Keywords",
        "N",
        "D",
        "G",
        "C",
        "A",
        "H",
        "M",
        "O",
        "anterior",
        "no fundus",
    ]

    left_df = pd.DataFrame(columns=["ID"])
    left_df["ID"] = df["ID"]
    left_df["Side"] = "L"
    left_df[newcols[2:5]] = df[cols[1:4]]
    left_df["Diagnostic Keywords"] = df["Left-Diagnostic Keywords"]
    left_df[newcols[6:-2]] = df[cols[15:23]]
    left_df[newcols[-2:]] = df[["L-ant", "L-no"]]

    right_df = pd.DataFrame(columns=["ID"])
    right_df["ID"] = df["ID"]
    right_df["Side"] = "R"
    right_df[newcols[2:5]] = df[[cols[i] for i in [1, 2, 4]]]
    right_df["Diagnostic Keywords"] = df["Right-Diagnostic Keywords"]
    right_df[newcols[6:-2]] = df[cols[23:-4]]
    right_df[newcols[-2:]] = df[["R-ant", "R-no"]]

    new_df = pd.concat([left_df, right_df], axis=0)
    new_df = new_df.sort_values(by=["ID", "Side"])
    new_df.to_csv(output_file, sep="\t", index=False, header=True)


if __name__ == "__main__":
    xsl_file = sys.argv[1]
    output_file = sys.argv[2]
    decode_d_k(xsl_file=xsl_file, output_file=output_file)