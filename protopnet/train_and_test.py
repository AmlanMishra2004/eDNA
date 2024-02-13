import time
import torch
import numpy as np
from sklearn import metrics

from helpers import list_of_distances, make_one_hot

np.set_printoptions(threshold=1000000)

def _train_or_test(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True,
                   coefs=None, scheduler=None, log=print):
    '''
    model: the multi-gpu model
    dataloader: sam: an iter() of a dataloader
    optimizer: if None, will be test evaluation
    '''
    # print(f"_train_or_test for dataloader of length {len(dataloader)}")
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_batches = 0
    total_cross_entropy = 0
    total_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0
    actual=[]
    prediction=[]
    for i, (sequence, label) in enumerate(dataloader):
        # print(f"i: {i}")
        # print(f"sequence: {sequence}")
        # print(f"label: {label}")
        input = sequence.cuda()
        target = label.cuda()
        actual.append(label.cpu().detach().numpy())

        # torch.enable_grad() has no effect outside of no_grad()
        grad_req = torch.enable_grad() if is_train else torch.no_grad()
        with grad_req:
            # nn.Module has implemented __call__() function
            # so no need to call .forward
            output, max_similarities = model(input)

            # compute loss
            cross_entropy = torch.nn.functional.cross_entropy(output, target)

            if class_specific:
                # max_dist = (model.prototype_shape[1]
                #             * model.prototype_shape[2])

                # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
                # calculate cluster cost
                prototypes_of_correct_class = torch.t(model.prototype_class_identity[:,label]).cuda()
                similarities, _ = torch.max((max_similarities) * prototypes_of_correct_class, dim=1)
                cluster_cost = torch.mean(similarities)

                # calculate separation cost
                prototypes_of_wrong_class = 1 - prototypes_of_correct_class
                similarities_to_nontarget_prototypes, _ = \
                    torch.max((max_similarities) * prototypes_of_wrong_class, dim=1)
                separation_cost = torch.mean(similarities_to_nontarget_prototypes)

                # calculate avg cluster cost
                avg_separation_cost = \
                    torch.sum(max_similarities * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)
                avg_separation_cost = torch.mean(avg_separation_cost)
                
                if use_l1_mask:
                    l1_mask = 1 - torch.t(model.prototype_class_identity).cuda()
                    l1 = (model.last_layer.weight * l1_mask).norm(p=1)
                else:
                    l1 = model.last_layer.weight.norm(p=1) 

            else:
                min_distance, _ = torch.min(max_similarities, dim=1)
                cluster_cost = torch.mean(min_distance)
                l1 = model.last_layer.weight.norm(p=1)

            # evaluation statistics
            _, predicted = torch.max(output.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()
            prediction.append(predicted.cpu().detach().numpy())
            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()

        # compute gradient and do SGD step
        if is_train:
            if class_specific:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['sep'] * separation_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost - 0.08 * separation_cost + 1e-4 * l1
            else:
                if coefs is not None:
                    loss = (coefs['crs_ent'] * cross_entropy
                          + coefs['clst'] * cluster_cost
                          + coefs['l1'] * l1)
                else:
                    loss = cross_entropy + 0.8 * cluster_cost + 1e-4 * l1
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

        del input
        del target
        del output
        del predicted
        del max_similarities

    end = time.time()
    
    # print(f"Prediction and type: {len(prediction)}, {type(prediction)}")
    # for arr in prediction:
    #     print(len(arr))
    prediction = prediction[:-1]
    # print(f"Actuals and type: {len(actual)}, {type(actual)}")
    # for ele in actual:
    #     print(len(ele))
    actual = actual[:-1]
    predictions=np.asarray(prediction)
    actuals=np.asarray(actual)
    predictions=predictions.flatten()
    actuals=actuals.flatten()
    # print(prediction)
    # print(type(prediction))
    # log('\ttime: \t{0}'.format(end -  start))
    # log('\tcross ent: \t{0}'.format(total_cross_entropy / n_batches))
    cross_ent = total_cross_entropy / n_batches
    # log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    cluster = total_cluster_cost / n_batches
    if class_specific:
        separation = total_separation_cost / n_batches
        avg_separation = total_avg_separation_cost / n_batches
    #     log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
    #     log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    # log('\taccu: \t\t{0}%'.format(n_correct / n_examples * 100))
    
    # log('\tl1: \t\t{0}'.format(model.last_layer.weight.norm(p=1).item()))
    # log('\t Calculating list_of_distances()')
    p = model.prototype_vectors.view(model.num_prototypes, -1).cpu()
    with torch.no_grad():
        p_avg_pair_dist = torch.mean(list_of_distances(p, p)).item()
    # log('\tp dist pair: \t{0}'.format(p_avg_pair_dist.item()))

    #confusiton matrix
    # confusion_mat = metrics.confusion_matrix(actuals, predictions)
    # log('\tconfusion matrix: \t\t\n{0}'.format(confusion_mat))
        
    ptype_results = {
        'time': end - start,
        'cross_ent': cross_ent,
        'cluster': cluster,
        'separation': separation,
        'avg_separation': avg_separation,
        'p_avg_pair_dist': p_avg_pair_dist
    }
    
    return actuals, predictions, ptype_results
    # return n_correct / n_examples, confusion_mat


def train(model, dataloader, optimizer, scheduler=None, class_specific=False,
          coefs=None, log=print):
    assert(optimizer is not None)
    
    # log('\ttrain')

    # for batch in dataloader:
        # print(f"First batch shape: {batch[0].shape}")
        # print("First example in first batch:")
        # print(batch[0][0])
        # break
    model.train()
    actual, predicted, ptype_results = _train_or_test(
        model=model,
        dataloader=dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        class_specific=class_specific,
        coefs=coefs,
        log=log
    )
    return actual, predicted, ptype_results


def test(model, dataloader, class_specific=False, log=print):
    # log('\ttest')
    model.eval()
    actual, predicted, ptype_results = _train_or_test(
        model=model,
        dataloader=dataloader,
        optimizer=None,
        class_specific=class_specific,
        log=log
    )
    return actual, predicted, ptype_results


def last_only(model, log=print):
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = False
    model.prototype_vectors.requires_grad = False
    for p in model.last_layer.parameters():
        p.requires_grad = True
    
    # log('\tlast layer')


def warm_only(model, log=print):
    for p in model.features.parameters():
        p.requires_grad = False
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True
    
    # log('\twarm')


def joint(model, log=print):
    for p in model.features.parameters():
        p.requires_grad = True
    for p in model.add_on_layers.parameters():
        p.requires_grad = True
    model.prototype_vectors.requires_grad = True
    for p in model.last_layer.parameters():
        p.requires_grad = True
    
    # log('\tjoint')