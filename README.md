# Multimodal-Single-Cell-Integration-Review

The goal of this assignment is to predict how DNA, RNA, and protein measurements co-vary in single cells as bone marrow stem cells develop into more mature blood cells. You will develop a model trained on a subset of 300,000-cell time course dataset of CD34+ hematopoietic stem and progenitor cells (HSPC) from four human donors at five time points generated for this competition by Cellarity, a cell-centric drug creation company.

In the test set, taken from an unseen later time point in the dataset, competitors will be provided with one modality and be tasked with predicting a paired modality measured in the same cell. 
<img width="705" alt="Screen Shot 2022-11-30 at 09 41 43" src="https://user-images.githubusercontent.com/113065896/204825271-06173c49-fa64-4d65-8228-2c4c46205441.png">

In the past decade, the advent of single-cell genomics has enabled the measurement of DNA, RNA, and proteins in single cells. These technologies allow the study of biology at an unprecedented scale and resolution. Among the outcomes have been detailed maps of early human embryonic development, the discovery of new disease-associated cell types, and cell-targeted therapeutic interventions. Moreover, with recent advances in experimental techniques it is now possible to measure multiple genomic modalities in the same cell.

While multimodal single-cell data is increasingly available, data analysis methods are still scarce. Due to the small volume of a single cell, measurements are sparse and noisy. Differences in molecular sampling depths between cells (sequencing depth) and technical effects from handling cells in batches (batch effects) can often overwhelm biological differences. When analyzing multimodal data, one must account for different feature spaces, as well as shared and unique variation between modalities and between batches. Furthermore, current pipelines for single-cell data analysis treat cells as static snapshots, even when there is an underlying dynamical biological process. Accounting for temporal dynamics alongside state changes over time is an open challenge in single-cell data science.

Generally, genetic information flows from DNA to RNA to proteins. DNA must be accessible (ATAC data) to produce RNA (GEX data), and RNA in turn is used as a template to produce protein (ADT data). These processes are regulated by feedback: for example, a protein may bind DNA to prevent the production of more RNA. This genetic regulation is the foundation for dynamic cellular processes that allow organisms to develop and adapt to changing environments. In single-cell data science, dynamic processes have been modeled by so-called pseudotime algorithms that capture the progression of the biological process. Yet, generalizing these algorithms to account for both pseudotime and real time is still an open problem.

--------------
#Main code
--------------
    import os, gc, scipy.sparse

    import pandas as pd

    import matplotlib.pyplot as plt

    import seaborn as sns

    import numpy as np

    from colorama import Fore, Back, Style

    from sklearn.decomposition import TruncatedSVD

    from matplotlib.ticker import PercentFormatter

    DATA_DIR = "/kaggle/input/open-problems-multimodal/"

    FP_CELL_METADATA = os.path.join(DATA_DIR,"metadata.csv")

    FP_CITE_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_cite_inputs.h5")

    FP_CITE_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_cite_targets.h5")

    FP_CITE_TEST_INPUTS = os.path.join(DATA_DIR,"test_cite_inputs.h5")

    FP_MULTIOME_TRAIN_INPUTS = os.path.join(DATA_DIR,"train_multi_inputs.h5")

    FP_MULTIOME_TRAIN_TARGETS = os.path.join(DATA_DIR,"train_multi_targets.h5")

    FP_MULTIOME_TEST_INPUTS = os.path.join(DATA_DIR,"test_multi_inputs.h5")

    FP_SUBMISSION = os.path.join(DATA_DIR,"sample_submission.csv")

    FP_EVALUATION_IDS = os.path.join(DATA_DIR,"evaluation_ids.csv")

--------------
#The Metadata table
--------------

The metadata table (which describes training and test data) shows us:

1. There is data about 281528 unique cells.
2. The cells belong to five days, four donors, eight cell types (including one type named 'hidden'), and two technologies.
3. The metadata table has no missing values.

        df_meta = pd.read_csv(FP_CELL_METADATA, index_col='cell_id')

        display(df_meta)

        if not df_meta.index.duplicated().any(): print('All cell_ids are unique.')

        if not df_meta.isna().any().any(): print('There are no missing values.')

<img width="358" alt="Screen Shot 2022-12-19 at 15 05 14" src="https://user-images.githubusercontent.com/113065896/208510708-2111c334-236a-4e66-9f74-1819084a082f.png">

Metadata distribution

    _, axs = plt.subplots(2, 2, figsize=(11, 6))

    for col, ax in zip(['day', 'donor', 'cell_type', 'technology'], axs.ravel()):

       vc = df_meta[col].astype(str).value_counts()
    
       if col == 'day':
    
           vc.sort_index(key = lambda x : x.astype(int), ascending=False, inplace=True)
        
       else:
    
           vc.sort_index(ascending=False, inplace=True)
        
       ax.barh(vc.index, vc, color=['MediumSeaGreen'])
    
       ax.set_ylabel(col)
    
       ax.set_xlabel('# cells')
    
     plt.tight_layout(h_pad=4, w_pad=4)

     plt.suptitle('Metadata distribution', y=1.04, fontsize=20)

     plt.show()

<img width="810" alt="Screen Shot 2022-12-19 at 15 06 47" src="https://user-images.githubusercontent.com/113065896/208510976-70d5454b-610d-4769-9e6d-e051b1af3eb0.png">

Explain: The CITEseq measurements took place on four days, the Multiome measurements on five (except that there are no measurements for donor 27678 on day 4. For every combination of day, donor and technology, there are around 8000 cells.

     df_meta_cite = df_meta[df_meta.technology=="citeseq"]
     df_meta_multi = df_meta[df_meta.technology=="multiome"]

     fig, axs = plt.subplots(1,2,figsize=(12,6))
     df_cite_cell_dist = df_meta_cite[["day","donor"]].value_counts().to_frame()\
                .sort_values("day").reset_index()\
                .rename(columns={0:"# cells"})
     sns.barplot(data=df_cite_cell_dist, x="day",hue="donor",y="# cells", ax=axs[0])
     axs[0].set_title(f"{len(df_meta_cite)} cells measured with CITEseq")

     df_multi_cell_dist = df_meta_multi[["day","donor"]].value_counts().to_frame()\
                .sort_values("day").reset_index()\
                .rename(columns={0:"# cells"})
     sns.barplot(data=df_multi_cell_dist, x="day",hue="donor",y="# cells", ax=axs[1])
     axs[1].set_title(f"{len(df_meta_multi)} cells measured with Multiome")
     plt.suptitle('# Cells per day, donor and technology', y=1.04, fontsize=20)
     plt.show()
     print('Average:', round(len(df_meta) / 35))
     
<img width="762" alt="Screen Shot 2022-12-19 at 15 12 42" src="https://user-images.githubusercontent.com/113065896/208512759-ac853324-b9a2-461f-afbd-ebc648e59284.png">

--------------
#The Time Series of the Cell Tyoes
--------------

In this assignment, we need to working with a time series. The cells in the given data can be classified into seven cell types. If we plot the ratio of cell types by day, we see that during the course of the experiment the hematopoietic stem cells (HSC, green in the diagram) slowly change into other cell types. We begin with 50 % HSC, and on day 7 only 20 % HSC remain. 

    daily_cell_types = df_meta_cite.groupby(['day', 'cell_type']).size().unstack()
    daily_cell_types[daily_cell_types.columns] = daily_cell_types.values / daily_cell_types.values.sum(axis=1).reshape(-1, 1)
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4), sharey=True)
    for cell_type in daily_cell_types.columns:
        ax1.plot(daily_cell_types.index,
                 daily_cell_types[cell_type],
                 label=cell_type,
                 lw = 7 if cell_type == 'HSC' else 3)
    ax1.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax1.set_xticks(daily_cell_types.index)
    ax1.legend()
    ax1.set_xlabel('Day')
    ax1.set_ylabel('Ratio')
    ax1.set_title('CITEseq ratio of cell types by day')

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    daily_cell_types = df_meta_multi.groupby(['day', 'cell_type']).size().unstack()
    daily_cell_types.drop(columns=['hidden'], inplace=True)
    daily_cell_types.drop(index=[10], inplace=True)
    daily_cell_types[daily_cell_types.columns] = daily_cell_types.values / daily_cell_types.values.sum(axis=1).reshape(-1, 1)
    daily_cell_types.loc[10] = [0.002, 0.128, 0.03, 0.33, 0.19, 0.05, 0.27] # estimate by visual extrapolation
    for i, cell_type in enumerate(daily_cell_types.columns):
        ax2.plot(daily_cell_types.index[:-1],
                 daily_cell_types[cell_type].iloc[:-1],
                 label=cell_type,
                 color=colors[i],
                 lw = 7 if cell_type == 'HSC' else 3)
        ax2.plot(daily_cell_types.index[-2:],
                 daily_cell_types[cell_type].iloc[-2:],
                 color=colors[i],
                 linestyle='dotted',
                 lw = 7 if cell_type == 'HSC' else 3)
    ax2.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax2.set_xticks([2,3,4,7,10])
    #ax2.legend()
    ax2.set_xlabel('Day')
    ax2.set_ylabel('Ratio')
    ax2.set_title('Multiome ratio of cell types by day')
    plt.show()

    del daily_cell_types
    
 <img width="810" alt="Screen Shot 2022-12-19 at 15 20 49" src="https://user-images.githubusercontent.com/113065896/208513960-fcda7ded-c1c9-4550-9625-63a316664e54.png">

--------------
#The CITEseq Inputs
--------------

The CITEseq input files contain 70988 samples (i.e., cells) for train and 48663 samples for test. 70988 + 48663 = 119651, which matches the number of rows in the CITEseq metadata table. No values are missing.

The input data corresponds to RNA expression levels for 22050 genes (there are 22050 columns).

The data have dtype float32, which means we need 119651 22050 4 = 10553218200 bytes = 10.6 GByte of RAM just for the features (train and test) without the targets.

Originally, these RNA expression levels were counts (i.e., nonnegative integers), but they have been normalized and log1p-transformed. With the log1p transformation, the data remain nonnegative.

Most columns have a minimum of zero, which means that for most genes there are cells which didn't express this gene (the count was 0). In fact, 78 % of all table entries are zero, and for some columns, the count is always zero.

    df_cite_train_x = pd.read_hdf(FP_CITE_TRAIN_INPUTS)
    display(df_cite_train_x.head())
    print('Shape:', df_cite_train_x.shape)
    print("Missing values:", df_cite_train_x.isna().sum().sum())
    print("Genes which never occur in train:", (df_cite_train_x == 0).all(axis=0).sum())
    print(f"Zero entries in train: {(df_cite_train_x == 0).sum().sum() / df_cite_train_x.size:.0%}")
    cite_gene_names = list(df_cite_train_x.columns)

<img width="626" alt="Screen Shot 2022-12-19 at 15 25 01" src="https://user-images.githubusercontent.com/113065896/208514638-fae44f24-00ce-4814-a20d-528e482c4e6f.png">

<img width="626" alt="Screen Shot 2022-12-19 at 15 25 20" src="https://user-images.githubusercontent.com/113065896/208514692-0fb098df-315e-4616-94d1-da720b05c545.png">

    plt.figure(figsize=(10, 4))
    plt.spy(df_cite_train_x[:5000])
    plt.show()

<img width="626" alt="Screen Shot 2022-12-19 at 15 25 53" src="https://user-images.githubusercontent.com/113065896/208514779-f76d58ac-001f-47f0-97b5-85996405e7e3.png">

Result: The histogram shows some artefacts because the data originally were integers. We don't show the zeros in the histogram, because with 78 % zeros, the histogram would have such a high peak at zero that we couldn't see anything else.

    nonzeros = df_cite_train_x.values.ravel()
    nonzeros = nonzeros[nonzeros != 0] # comment this line if you want to see the peak at zero
    plt.figure(figsize=(16, 4))
    plt.gca().set_facecolor('#0057b8')
    plt.hist(nonzeros, bins=500, density=True, color='#ffd700')
    print('Minimum nonzero value:', nonzeros.min())
    del nonzeros
    plt.title("Histogram of nonzero RNA expression levels in train")
    plt.xlabel("log1p-transformed expression count")
    plt.ylabel("density")
    plt.show()
    
<img width="807" alt="Screen Shot 2022-12-19 at 15 27 32" src="https://user-images.githubusercontent.com/113065896/208515074-dd777cfb-bebe-47b9-9e09-23feb39d254b.png">

    _, axs = plt.subplots(5, 4, figsize=(16, 16))
    for col, ax in zip(df_cite_train_x.columns[:20], axs.ravel()):
        nonzeros = df_cite_train_x[col].values
        nonzeros = nonzeros[nonzeros != 0] # comment this line if you want to see the peak at zero
        ax.hist(nonzeros, bins=100, density=True)
        ax.set_title(col)
    plt.tight_layout(h_pad=2)
    plt.suptitle('Histograms of nonzero RNA expression levels for selected features', fontsize=20, y=1.04)
    plt.show()
    del nonzeros
    
![image](https://user-images.githubusercontent.com/113065896/208515184-ee6f7bf7-b3b3-43f9-b7c6-e76d3c7d9f3d.png)

    cell_index = df_cite_train_x.index
    meta = df_meta_cite.reindex(cell_index)
    gc.collect()
    df_cite_train_x = scipy.sparse.csr_matrix(df_cite_train_x.values)

Comment: So far, We have freed enough memory to analyze the test data; afterwards we convert the test data to a CSR matrix as well.

    df_cite_test_x = pd.read_hdf(FP_CITE_TEST_INPUTS)
    print('Shape of CITEseq test:', df_cite_test_x.shape)
    print("Missing values:", df_cite_test_x.isna().sum().sum())
    print("Genes which never occur in test: ", (df_cite_test_x == 0).all(axis=0).sum())
    print(f"Zero entries in test:  {(df_cite_test_x == 0).sum().sum() / df_cite_test_x.size:.0%}")


    gc.collect()
    cell_index_test = df_cite_test_x.index
    meta_test = df_meta_cite.reindex(cell_index_test)
    df_cite_test_x = scipy.sparse.csr_matrix(df_cite_test_x.values)
    
<img width="338" alt="Screen Shot 2022-12-19 at 15 32 16" src="https://user-images.githubusercontent.com/113065896/208515881-b667e559-1664-49e1-a223-28a942c8d25e.png">

--------------
#The Distributions of Train and Test in Feature Space
--------------
For the next diagrams, we need to project the data (train and test together) to two dimensions. Usually PCA, but the scikit-learn PCA implementation needs too much memory. Fortunately, TruncatedSVD does a similar projection and uses much less memory:

    # Concatenate train and test for the SVD
    both = scipy.sparse.vstack([df_cite_train_x, df_cite_test_x])
    print(f"Shape of both before SVD: {both.shape}")

    # Project to two dimensions
    svd = TruncatedSVD(n_components=2, random_state=1)
    both = svd.fit_transform(both)
    print(f"Shape of both after SVD:  {both.shape}")

    # Separate train and test
    X = both[:df_cite_train_x.shape[0]]
    Xt = both[df_cite_train_x.shape[0]:]

<img width="338" alt="Screen Shot 2022-12-19 at 15 36 14" src="https://user-images.githubusercontent.com/113065896/208516617-164aecd7-e641-4c72-831e-ea74c475e728.png">

Preview: The scatterplots below show the extent of the data for every day and every donor (SVD projection to two dimensions). The nine diagrams with orange dots make up the training data. The three diagrams with orange-red dots below are the public test set and the four dark red diagrams at the right are the private test set.

The gray area marks the complete training data (union of the nine orange diagrams), and the black area behind marks the test data.

We see that the distributions differ. In particular the day 2 distribution (left column of diagrams) is much less wide than the others.

    # Scatterplot for every day and donor
    _, axs = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(12, 11))
    for donor, axrow in zip([13176, 31800, 32606, 27678], axs):
        for day, ax in zip([2, 3, 4, 7], axrow):
            ax.scatter(Xt[:,0], Xt[:,1], s=1, c='k')
            ax.scatter(X[:,0], X[:,1], s=1, c='lightgray')
            if day != 7 and donor != 27678: # train
                temp = X[(meta.donor == donor) & (meta.day == day)]
                ax.scatter(temp[:,0], temp[:,1], s=1, c='orange')
            else: # test
                temp = Xt[(meta_test.donor == donor) & (meta_test.day == day)]
                ax.scatter(temp[:,0], temp[:,1], s=1, c='darkred' if day == 7 else 'orangered')
            ax.set_title(f'Donor {donor} day {day}')
            ax.set_aspect('equal')
    plt.suptitle('CITEseq features, projected to the first two SVD components', y=0.95, fontsize=20)
    plt.show()

![image](https://user-images.githubusercontent.com/113065896/208516795-3bb2d891-c996-4f0c-b24d-1368b2bf7d43.png)

    df_cite_train_x, df_cite_test_x, X, Xt = None, None, None, None # release the memory
    
--------------
#CITEseq targets (surface protein levels)
--------------
The CITEseq output (target) file is much smaller: it has 70988 rows like the training input file, but only 140 columns. The 140 columns correspond to 140 proteins.

The targets are dsb-normalized surface protein levels. We plot the histograms of a few selected columns and see that the distributions vary: some columns are normally distributed, some columns are multimodal, some have other shapes, and there seem to be outliers.

    df_cite_train_y = pd.read_hdf(FP_CITE_TRAIN_TARGETS)
    display(df_cite_train_y.head())
    print('Output shape:', df_cite_train_y.shape)

    _, axs = plt.subplots(5, 4, figsize=(16, 16))
    for col, ax in zip(['CD86', 'CD270', 'CD48', 'CD8', 'CD7', 'CD14', 'CD62L', 'CD54', 'CD42b', 'CD2', 'CD18', 'CD36', 'CD328', 'CD224', 'CD35', 'CD57', 'TCRVd2', 'HLA-E', 'CD82', 'CD101'], axs.ravel()):
        ax.hist(df_cite_train_y[col], bins=100, density=True)
        ax.set_title(col)
    plt.tight_layout(h_pad=2)
    plt.suptitle('Selected target histograms (surface protein levels)', fontsize=20, y=1.04)
    plt.show()

    cite_protein_names = list(df_cite_train_y.columns)

<img width="664" alt="Screen Shot 2022-12-19 at 15 40 05" src="https://user-images.githubusercontent.com/113065896/208517280-34b45ccd-2c4e-4da1-9bff-d32c92ee2870.png">

![image](https://user-images.githubusercontent.com/113065896/208517335-7d2fad83-b437-492b-9fd1-334a3df5d9ef.png)

A projection of the CITEseq targets to two dimensions again shows that the groups have different distributions.

    svd = TruncatedSVD(n_components=2, random_state=1)
    X = svd.fit_transform(df_cite_train_y)

    # Scatterplot for every day and donor
    _, axs = plt.subplots(4, 4, sharex=True, sharey=True, figsize=(12, 11))
    for donor, axrow in zip([13176, 31800, 32606, 27678], axs):
        for day, ax in zip([2, 3, 4, 7], axrow):
            if day != 7 and donor != 27678: # train
                ax.scatter(X[:,0], X[:,1], s=1, c='lightgray')
                temp = X[(meta.donor == donor) & (meta.day == day)]
                ax.scatter(temp[:,0], temp[:,1], s=1, c='orange')
            else: # test
                ax.text(50, -25, '?', fontsize=100, color='gray', ha='center')
            ax.set_title(f'Donor {donor} day {day}')
            ax.set_aspect('equal')
    plt.suptitle('CITEseq target, projected to the first two SVD components', y=0.95, fontsize=20)
    plt.show()
    
![image](https://user-images.githubusercontent.com/113065896/208517467-63140d72-77f1-476c-bf6a-5c0862c92771.png)

    df_cite_train_x, df_cite_train_y, X, svd = None, None, None, None # release the memory
    
--------------
#Name Matching
--------------
The CITEseq task has genes as input and proteins as output. Genes encode proteins, and it is more or less known which genes encode which proteins. This information is encoded in the column names: The input dataframe has the genes as column names, and the target dataframe has the proteins as column names. According to the naming convention, the gene names contain the protein name as suffix after a '_'.

If we match the input column names with the target column names, we find 151 genes which encode a target protein (see the table below). It doesn't matter that some proteins are encoded by more than one gene (e.g., rows 146 and 147 of the table). We may assume that these 151 features will have a high feature importance in our models.

    matching_names = []
    for protein in cite_protein_names:
        matching_names += [(gene, protein) for gene in cite_gene_names if protein in gene]
    pd.DataFrame(matching_names, columns=['Gene', 'Protein'])

<img width="261" alt="Screen Shot 2022-12-19 at 15 42 59" src="https://user-images.githubusercontent.com/113065896/208517751-00ee6020-caf2-4a1d-b05f-ea6eecdf46a9.png">

--------------
#Multiome input
--------------
he Multiome dataset is much larger than the CITEseq part and way too large to fit into 16 GByte RAM:

1. train inputs: 105942 * 228942 float32 values (97 GByte)
2. train targets: 105942 * 23418 float32 values (10 GByte)
3. test inputs: 55935 * 228942 float32 values (13 GByte)

For this EDA, we read all the data to check for missing, zero and negative values, we plot a histogram and we look at the Y chromosome, but we don't analyze much more.

No values are missing.

The data consists of ATAC-seq peak counts transformed with TF-IDF. They are all nonnegative. In the sample we are looking at, 98 % of the entries are zero.

    bins = 100
    cell_summary = pd.DataFrame()

    def analyze_multiome_x(filename):
        global cell_summary
        start = 0
        chunksize = 5000
        total_rows = 0
        maximum_x = 0

        while True: # read the next chunk of the file
            X = pd.read_hdf(filename, start=start, stop=start+chunksize)
            if X.isna().any().any(): print('There are missing values.')
            if (X < 0).any().any(): print('There are negative values.')
            total_rows += len(X)
            print(total_rows, 'rows read')

            donors = df_meta_multi.donor.reindex(X.index) # metadata: donor of cell
            days = df_meta_multi.day.reindex(X.index) # metadata: day of cell
            chrY_cols = [f for f in X.columns if 'chrY' in f]
            maximum_x = max(maximum_x, X[chrY_cols].values.ravel().max())
            for donor in [13176, 31800, 32606, 27678]:
                hist, _ = np.histogram(X[chrY_cols][donors == donor].values.ravel(), bins=bins, range=(0, 15))
                chrY_histo[donor] += hist

            cell_summary = pd.concat([cell_summary,
                                      pd.DataFrame({'donor': donors,
                                                    'day': days,
                                                    'total': X.sum(axis=1),
                                                    'total_nonzero': (X != 0).sum(axis=1)})])
            if len(X) < chunksize: break
            start += chunksize

        display(X.head(3))
        print(f"Zero entries in {filename}: {(X == 0).sum().sum() / X.size:.0%}")

    chrY_histo = dict()
    for donor in [13176, 31800, 32606, 27678]:
        chrY_histo[donor] = np.zeros((bins, ), int)

    # Look at the training data
    analyze_multiome_x(FP_MULTIOME_TRAIN_INPUTS)
    
<img width="732" alt="Screen Shot 2022-12-19 at 15 55 28" src="https://user-images.githubusercontent.com/113065896/208522279-d9af0f24-841c-48fe-9a6b-a722a8c86ffa.png">

<img width="732" alt="Screen Shot 2022-12-19 at 15 45 16" src="https://user-images.githubusercontent.com/113065896/208518121-2137c048-c16e-4f8a-82c0-7360c593622a.png">

Comment: In the histogram, we again hide the peak for the 98 % zero values.

    df_multi_train_x = pd.read_hdf(FP_MULTIOME_TRAIN_INPUTS, start=0, stop=5000)
    nonzeros = df_multi_train_x.values.ravel()
    nonzeros = nonzeros[nonzeros != 0] # comment this line if you want to see the peak at zero
    plt.figure(figsize=(16, 4))
    plt.gca().set_facecolor('#0057b8')
    plt.hist(nonzeros, bins=500, density=True, color='#ffd700')
    del nonzeros
    plt.title("Histogram of nonzero feature values (subset)")
    plt.xlabel("TFIDF-transformed peak count")
    plt.ylabel("density")
    plt.show()

    del df_multi_train_x # free the memory

![image](https://user-images.githubusercontent.com/113065896/208518213-f9c5df6a-9f5b-492a-bfad-ee0562138e51.png)

And the, We can do the same checks for the test data:

    # Look at the test data
    analyze_multiome_x(FP_MULTIOME_TEST_INPUTS)
    
<img width="732" alt="Screen Shot 2022-12-19 at 15 56 09" src="https://user-images.githubusercontent.com/113065896/208522399-fce4afaf-ee9c-4684-89d6-8a5730dfbc0a.png">

<img width="731" alt="Screen Shot 2022-12-19 at 15 47 32" src="https://user-images.githubusercontent.com/113065896/208519208-369fcd96-f7c7-4f03-8d97-10b5c566a704.png">

--------------
#Batch Effects
--------------
Biological experiments are notorious for batch effects: If you repeat the same experiment several times, you get systematic differences in the measurements. We can show these batch effects for the ATACseq measurements. Because of the huge data size, we look only at the row sums of the data (i.e. the sum of all measured values of a cell). In the diagram, we plot these sums for every row of the train and test datasets, and we color the dots by the day of the experiment (day 2 is red, day 3 is green and so on).

The diagram clearly shows that day 3 (green) gave the highest measurements, day 7 (yellow) the lowest. There are differences among donors as well, albeit smaller ones.

    def plot_batch_effects():
        _, axs = plt.subplots(1, 2, sharey=True, figsize=(12, 10))
        color = cell_summary.day.map({2: 'r', 3: 'g', 4: 'b', 7: 'y', 10: 'gray'})
        axs[0].scatter(cell_summary.total_nonzero, np.arange(len(cell_summary)), s=0.1, c=color)
        axs[0].set_xlabel('cell total')
        axs[1].scatter(cell_summary.total, np.arange(len(cell_summary)), s=0.1, c=color)
        axs[1].set_xlabel('cell total nonzeros')
        axs[0].set_ylabel('cell')
        axs[0].invert_yaxis()
        plt.suptitle('Row totals colored by day', y=0.94, fontsize=20)
        plt.show()

    plot_batch_effects()

![image](https://user-images.githubusercontent.com/113065896/208519949-0a0779a2-0c65-4485-ba8b-ef0ff5760371.png)

--------------
#Y chromosome
--------------

    plt.rcParams['savefig.facecolor'] = "1.0"
    _, axs = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(14, 4))
    for donor, ax in zip([13176, 31800, 32606, 27678], axs):
        ax.set_title(f"Donor {donor} {'(test)' if donor == 27678 else ''}", fontsize=16)
        total = chrY_histo[donor].sum()
        ax.fill_between(range(bins-1), chrY_histo[donor][1:] / total, color='limegreen')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.suptitle("Histogram of nonzero Y chromosome accessibility", y=0.95, fontsize=20)
    plt.tight_layout()
    plt.show()
    
![image](https://user-images.githubusercontent.com/113065896/208520397-98db0a25-0052-4fa5-a9ba-9ba79f5315a7.png)

Comment: The histograms of the Y chromosomes illustrate the diversity of the donors: Donor 13176 seems to have (almost) no Y chromosome (maybe the few nonzero values are measuring errors). It appears that the donors are one woman and three men.

--------------
#Multiome target
--------------
The Multiome targets (RNA count data) are in similar shape as the CITEseq inputs: They have 105942 rows and 23418 columns. All targets are nonnegative and no values are missing.

    cell_summary = pd.DataFrame()
    start = 0
    chunksize = 10000
    total_rows = 0
    while True:
        df_multi_train_y = pd.read_hdf(FP_MULTIOME_TRAIN_TARGETS, start=start, stop=start+chunksize)
        if df_multi_train_y.isna().any().any(): print('There are missing values.')
        if (df_multi_train_y < 0).any().any(): print('There are negative values.')
        total_rows += len(df_multi_train_y)
        print(total_rows, 'rows read')

        donors = df_meta_multi.donor.reindex(df_multi_train_y.index) # metadata: donor of cell
        days = df_meta_multi.day.reindex(df_multi_train_y.index) # metadata: day of cell
        cell_summary = pd.concat([cell_summary,
                                  pd.DataFrame({'donor': donors,
                                                'day': days,
                                                'total': df_multi_train_y.sum(axis=1),
                                                'total_nonzero': (df_multi_train_y != 0).sum(axis=1)})])

        if len(df_multi_train_y) < chunksize: break
        start += chunksize

    display(df_multi_train_y.head())
    
<img width="721" alt="Screen Shot 2022-12-19 at 15 50 35" src="https://user-images.githubusercontent.com/113065896/208521269-2be978af-b3c4-47bf-b934-0da2a1c04b1a.png">

<img width="721" alt="Screen Shot 2022-12-19 at 15 50 50" src="https://user-images.githubusercontent.com/113065896/208521421-eccda6be-fad4-4ec7-b920-4a455efe2a1b.png">

    nonzeros = df_multi_train_y.values.ravel()
    nonzeros = nonzeros[nonzeros != 0]
    plt.figure(figsize=(16, 4))
    plt.gca().set_facecolor('#0057b8')
    plt.hist(nonzeros, bins=500, density=True, color='#ffd700')
    del nonzeros
    plt.title("Histogram of nonzero target values (based on a subset of the rows)")
    plt.xlabel("log1p-transformed expression count")
    plt.ylabel("density")
    plt.show()

    df_multi_train_y = None # release the memory
    
![image](https://user-images.githubusercontent.com/113065896/208521561-69009e3c-3a43-426b-94db-e273cabbd47d.png)
    
<img width="721" alt="Screen Shot 2022-12-19 at 15 51 23" src="https://user-images.githubusercontent.com/113065896/208521589-4b465a03-95d7-48c4-b488-d258ea5e8538.png">

Comment: Above, we saw the batch effects in the ATACseq data; below we see that the gene expression dataset has its own, independent, batch effects: For the gene expression data, day 7 (yellow) clearly has the lowest totals. Day 3 (green), which deviated the most from the other days for the ATACseq data, is not far from the mean for the gene expressions. But again, differences among days seem to exceed differences among donors.

The good news here is that the competition metric does not depend on shifts in the target values: Even if for day 7 (and maybe for day 10) all measurements are shifted by a constant, the predictions can still achieve a good correlation with the true values. With a mean-squared-error metric, this would be different.

    plot_batch_effects()
    
![image](https://user-images.githubusercontent.com/113065896/208521709-4a3d4502-4f06-46a3-96dd-3fa119e53989.png)
