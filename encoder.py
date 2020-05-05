    figures_dir = "/home/henrik/PycharmProjects/vae_for_retinal_images/data/supervised"
    encoder_name = "deep_balanced"
    os.makedirs(figures_dir + f'/{encoder_name}', exist_ok=True)

    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    # torch.cuda.clear_memory_allocated()
    torch.cuda.empty_cache()
    # torch.cuda.memory_stats(device)

    csv_df = pd.read_csv(csv_file, sep='\t')

    diagnoses = {
        "N": "normal fundus",
        "D": "proliferative retinopathy",
        "G": "glaucoma",
        "C": "cataract",
        "A": "age related macular degeneration",
        "H": "hypertensive retinopathy",
        "M": "myopia",
        "O": "other diagnosis"
    }
    number_of_diagnoses = len(diagnoses)
    diagnoses_list = list(diagnoses.keys())
    diagnoses_list.extend(["Patient Sex"])
    angles = [x for x in range(-22, -9)]
    angles.extend([x for x in range(10, 22 + 1)])
    angles.extend([x for x in range(-9, 10)])
    print("\nPossible Angles: {}\n".format(angles))

    print("\nLoad Data as Tensors and build targets simultanously...")
    targets = []
    data = []
    marker = None

    for i, jpg in tqdm(enumerate(os.listdir(trainfolder))):
        if jpg == '.snakemake_timestamp':
            marker = True
            continue

        data.append(io.imread(trainfolder + jpg).transpose((2, 0, 1)))

        jpg = jpg.replace("_flipped", "")
        for angle in angles:
            jpg = jpg.replace("_rot_%i" % angle, "")
        row_number = csv_df.loc[csv_df['Fundus Image'] == jpg].index[0]

        targets_for_img = np.zeros((number_of_diagnoses+1))
        for j, feature in enumerate(diagnoses_list):
            if not marker:
                if feature == "Patient Sex":
                    targets_for_img[j] = 0 if csv_df.iloc[row_number].at[feature] == "Female" else 1
                else:
                    targets_for_img[j] = csv_df.iloc[row_number].at[feature]
            else:
                if feature == "Patient Sex":
                    targets_for_img[j] = 0 if csv_df.iloc[row_number].at[feature] == "Female" else 1
                else:
                    targets_for_img[j] = csv_df.iloc[row_number].at[feature]

        targets.append(targets_for_img)

    data = torch.Tensor(data)
    targets = torch.Tensor(targets).detach()   #.float()
    print("\nSize of the dataset: {}\nShape of the single tensors: {}".format(data.size(0), data[0].shape))

    data_size = data.size(0)
    net = Encoder(number_of_features=len(diagnoses_list)).to(device=device)
    # print("Allocated memory: %s MiB" % torch.cuda.memory_allocated(device))

    # Train the network
    n_epochs = 1
    learning_rate = 5e-5
    criterion = nn.BCELoss().to(device=device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    lossarray = []

    # calculate batch_size
    batch_size = calc_batch_size(data_size, batch_size=80)

    # Train network
    start = time.perf_counter()

    d_mod_b = data_size % batch_size

    print("Start Training")
    for n in tqdm(range(n_epochs)):
        running_loss = 0.0
        b_size = batch_size

        for i in range(0, data_size, batch_size):
            if (i + batch_size) > data_size and d_mod_b != 1:
                # for uncompleted last batch
                b_size = d_mod_b

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(data[i:(i+b_size)].to(device))
            loss = criterion(outputs, targets[i:(i+b_size)].to(device))
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if 1: # i % b_size == b_size - 1:
                print('[%d, %5d] loss: %.3f' % (n + 1, i + 1, running_loss / batch_size))
                lossarray.append(loss.item())
                running_loss = 0.0

    print('Finished Training\nTrainingtime: %d sec' % (time.perf_counter() - start))
    print(lossarray)
    x = np.arange(len(lossarray))
    spl = UnivariateSpline(x, lossarray)
    plt.title("Loss-Curve", fontsize=16, fontweight='bold')
    plt.plot(x, lossarray, '-y')
    plt.plot(x, spl(x), '-r')
    plt.savefig(f'{figures_dir}/{encoder_name}_loss_curve.png')
    # plt.show()
    plt.close()

    PATH = f'{figures_dir}/{encoder_name}/{encoder_name}.pth'
    torch.save(net.state_dict(), PATH)

    ########################################
    #           Test network               #
    ########################################
    # torch.cuda.clear_memory_allocated()
    torch.cuda.empty_cache()
    # torch.cuda.memory_stats(device)

    print("\nLoad Data as Tensors and build targets simultanously...")
    targets = []
    data = []
    marker = None

    for i, jpg in tqdm(enumerate(os.listdir(testfolder))):
        if jpg == '.snakemake_timestamp':
            marker = True
            continue

        data.append(io.imread(testfolder + jpg).transpose((2, 0, 1)))

        jpg = jpg.replace("_flipped", "")
        for angle in angles:
            jpg = jpg.replace("_rot_%i" % angle, "")
        row_number = csv_df.loc[csv_df['Fundus Image'] == jpg].index[0]

        targets_for_img = np.zeros((number_of_diagnoses + 1))
        for j, feature in enumerate(diagnoses_list):
            if not marker:
                if feature == "Patient Sex":
                    targets_for_img[j] = 0 if csv_df.iloc[row_number].at[feature] == "Female" else 1
                else:
                    targets_for_img[j] = csv_df.iloc[row_number].at[feature]
            else:
                if feature == "Patient Sex":
                    targets_for_img[j] = 0 if csv_df.iloc[row_number].at[feature] == "Female" else 1
                else:
                    targets_for_img[j] = csv_df.iloc[row_number].at[feature]

        targets.append(targets_for_img)

    data = torch.Tensor(data).detach()
    targets = torch.Tensor(targets).detach() # .float()
    data_size = data.size(0)

    # Test the network
    print("Start testing the network..")
    batch_size = calc_batch_size(data_size, batch_size=60)

    print("Allocated memory: %s MiB" % torch.cuda.memory_allocated(device))
    print("Make predictions...")
    outputs = torch.zeros((data_size, number_of_diagnoses + 1), device=device).detach()
    for i in range(0, data_size, batch_size):
        # for uncompleted last batch
        if (i + batch_size) > data_size and d_mod_b != 1:
            batch_size = d_mod_b

        outputs[i:(i + batch_size)] = net(data[i:(i + batch_size)].to(device))


    # To measure the accuracy on the basic of the rounded outcome for each diagnosis could lead to a less
    # meaningful result. That's why this approach is deprecated.
    # In lieu thereof, a ROC and PR curve is used.
    # The network has as an outcome a vector of floats with values between 0 and 1. The threshold to round up is
    # increased stepwise, starts with 0 until 1.
    # In every step we calculate the Sensitivity/True Positiv Rate (TRP) and the False Positive Rate (1-Specifity):
    # TRP = TP/(TP+FN)  and  FPR=FP/(TN+FP).
    # https://de.wikipedia.org/wiki/Beurteilung_eines_bin%C3%A4ren_Klassifikators#Sensitivit%C3%A4t_und_Falsch-Negativ-Rate
    # https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc

    # roc_auc_score: Compute Area Under the Receiver Operating Characteristic Curve (ROC AUC) from prediction scores:
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html

    # average_auc_score: Compute average precision (AP) from prediction scores
    # AP = sum ((R_N - R_N_-1) * P_N)
    # AP summarizes a precision-recall curve as the weighted mean of precisions achieved at each threshold, with the
    # increase in recall from the previous threshold used as the weight: where R_N and P_N are the precision and recall
    # at the n-th threshold. This implementation is not interpolated and is different from computing the area under the
    # precision-recall curve with the trapezoidal rule, which uses linear interpolation and can be too optimistic.
    # Recall = TP/TP+FN  and   Precision = TP/TP+FP
    # F1 Score = 2*(Recall * Precision) / (Recall + Precision)
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html#sklearn.metrics.average_precision_score
    # https://machinelearningmastery.com/roc-curves-and-precision-recall-curves-for-classification-in-python/

    # ROC-Curve/AUC with sklearn:
    from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, average_precision_score

    outputs = outputs.to(device="cpu").detach().numpy()
    targets = targets.float().numpy()
    colors = ['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'indigo', 'darkgreen', 'firebrick', 'sienna',
              'red', 'limegreen']

    tpr = dict()  # Sensitivity/False Positive Rate
    fpr = dict()   # True Positive Rate / (1-Specifity)
    auc = dict()

    # A "micro-average": quantifying score on all classes jointly
    tpr["micro"], fpr["micro"], _ = roc_curve(targets.ravel(), outputs.ravel())
    auc["micro"] = roc_auc_score(targets.ravel(), outputs.ravel(), average='micro')
    print('AUC score, micro-averaged over all classes: {0:0.2f}'.format(auc['micro']))

    plt.figure()
    plt.step(tpr['micro'], fpr['micro'], where='post')
    plt.xlabel('False Positive Rate / Sensitivity', fontsize=11)
    plt.ylabel('True Negative Rate / (1 - Specifity)', fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'AUC score, micro-averaged over all classes: AP={0:0.2f}'
            .format(auc["micro"]), fontsize=13, fontweight='bold')
    plt.show()
    plt.savefig(f'{figures_dir}/{encoder_name}/ROC_curve_micro_averaged.png')
    plt.close()

    # Plot of all classes ('macro')
    for i in range(number_of_diagnoses + 1):
        tpr[i], fpr[i], _ = roc_curve(targets[:, i], outputs[:, i])
        try:
            auc[i] = roc_auc_score(targets[:, i], outputs[:, i])
        except ValueError:
            print(i, diagnoses_list[i], targets[:,i], outputs[:,i])

    plt.figure(figsize=(7, 9))
    lines = []
    labels = []

    l, = plt.plot(tpr["micro"], fpr["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-averaged ROC-AUC = {0:0.2f})'.format(auc["micro"]))

    for i, color in zip(range(number_of_diagnoses + 1), colors):
        if i in auc.keys():
            l, = plt.plot(tpr[i], fpr[i], color=color, lw=0.5)
            lines.append(l)
            if diagnoses_list[i] != "Patient Sex":
                labels.append('ROC for class {0} (ROC-AUC = {1:0.2f})'
                              ''.format(diagnoses[diagnoses_list[i]], auc[i]))
            else:
                labels.append('ROC for class {0} (ROC-AUC = {1:0.2f})'
                              ''.format(diagnoses_list[i], auc[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlabel('False Positive Rate / Sensitivity', fontsize=11)
    plt.ylabel('True Negative Rate / (1 - Specifity)', fontsize=11)
    plt.title('ROC curve of all features', fontsize=13, fontweight='bold')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=8))
    plt.savefig(f'{figures_dir}/{encoder_name}/ROC_curve_of_all_features.png')
    plt.show()
    plt.close()

    # Precision-Recall Plots
    precision = dict()
    recall = dict()
    average_precision = dict()
    from sklearn.metrics import auc

    # A "micro-average": quantifying score on all classes jointly
    precision["micro"], recall["micro"], _ = precision_recall_curve(targets.ravel(), outputs.ravel())
    average_precision["micro"] = average_precision_score(targets, outputs, average="micro")
    print('Average precision score, micro-averaged over all classes: {0:0.2f}'.format(average_precision["micro"]))
    plt.figure()
    plt.step(recall['micro'], precision['micro'], where='post')
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(
        'Average precision score, micro-averaged over all classes: AP={0:0.2f}'
            .format(average_precision["micro"]), fontsize=13, fontweight='bold')
    plt.show()
    plt.savefig(f'{figures_dir}/{encoder_name}/PR_curve_micro_averaged.jpg')
    plt.close()

    # Plot of all classes ('macro')
    for i in range(number_of_diagnoses + 1):
        precision[i], recall[i], _ = precision_recall_curve(targets[:, i], outputs[:, i])
        average_precision[i] = average_precision_score(targets[:, i], outputs[:, i])

    plt.figure(figsize=(7, 9))
    f_scores = np.linspace(0.2, 0.8, num=4)
    lines = []
    labels = []
    for f_score in f_scores:
        x = np.linspace(0.01, 1)
        y = f_score * x / (2 * x - f_score)
        l, = plt.plot(x[y >= 0], y[y >= 0], color='gray', alpha=0.2)
        plt.annotate('f1={0:0.1f}'.format(f_score), xy=(0.9, y[45] + 0.02))

    lines.append(l)
    labels.append('iso-f1 curves')
    l, = plt.plot(recall["micro"], precision["micro"], color='gold', lw=2)
    lines.append(l)
    labels.append('micro-average Precision-recall (average precision = {0:0.2f}'
                  ''.format(average_precision["micro"]))

    for i, color in zip(range(number_of_diagnoses + 1), colors):
        l, = plt.plot(recall[i], precision[i], color=color, lw=0.5)
        lines.append(l)
        if diagnoses_list[i] != "Patient Sex":
            labels.append('Precision-recall for class {0} (AP = {1:0.2f})'
                          ''.format(diagnoses[diagnoses_list[i]], average_precision[i]))
        else:
            labels.append('Precision-recall for class {0} (AP = {1:0.2f})'
                          ''.format(diagnoses_list[i], average_precision[i]))

    fig = plt.gcf()
    fig.subplots_adjust(bottom=0.25)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=11)
    plt.ylabel('Precision', fontsize=11)
    plt.title('Precision-Recall curve of all features')
    plt.legend(lines, labels, loc=(0, -.38), prop=dict(size=9))
    plt.show()
    plt.savefig(f'{figures_dir}/{encoder_name}/PR_curve_of_all_features.jpg')

    os.system(f"cp encoder.py {figures_dir}/{encoder_name}/encoder.py")
