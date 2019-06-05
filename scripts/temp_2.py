# loop over large fragments of large_ws len (150k)
for sample_idx in range(large_x.size(0)):
    # large_x[sample_idx].size() must be (bs_x, 1, small_ws)
    # Z = enc_embeds.size() must be (bs_x, ch_out, len_out)
    enc_embeds_one = model_enc.forward(large_x[sample_idx])

    if sample_idx == 0:
        enc_embeds = enc_embeds_one.clone()
    else:
        enc_embeds = torch.cat((enc_embeds, enc_embeds_one), dim=0)

ar_out = model_ar.forward(enc_embeds)



loss = 0
# loop over ...
for z_idx in range(1, enc_embeds.size(0) - 1):
    c = enc_embeds[:z_idx].view(1, enc_embeds.size(0), enc_embeds.size(1))
    ar_c = model_ar.forward(c)




    # loop over large fragments of len == large_ws
    for x_idx in range(large_x.shape[0]):
        # large_x[x_idx].size() must be (bs_x, 1, small_ws)
        # Z = enc_embeds.size() must be (bs_x, ch_out * len_out)
        enc_embeds_single = models_dict['enc'].forward(large_x[x_idx])
        if x_idx == 0:
            enc_embeds = torch.empty(0,
                                     enc_embeds_single.shape[0],
                                     enc_embeds_single.shape[1],
                                     device=cuda)
        enc_embeds = torch.cat((enc_embeds, enc_embeds_single), dim=0)
    del enc_embeds_single
# c_w, target_bin_pred = models_dict['pred'].forward(c_t)
# z_preds = torch.matmul(c_w, z_next.t()).exp()

    logits = F.log_softmax(z_preds, dim=1)
    loss1 += logits.diag().mean()
    loss2 += F.cross_entropy(target_bin_pred, target_bin[x_idx, :-1])

loss = loss1 + loss2
loss.backward()
optimizer.step()

# enc_norm = calc_grad_norm(models_dict['enc'])
# ar_norm = calc_grad_norm(models_dict['ar'])
# target_head_norm = calc_grad_norm(models_dict['target_head'])



# # from validation
# loss1 = torch.tensor(0.0, device=cuda)
# loss2 = torch.tensor(0.0, device=cuda)
# # loop over large fragments of len == large_ws
# for x_idx in range(large_x.size(0)):
#     # large_x[x_idx].size() must be (bs_x, 1, small_ws)
#     # Z = enc_embeds.size() must be (bs_x, ch_out * len_out)
#     enc_embeds = models_dict['enc'].forward(large_x[x_idx])
#
#     c = models_dict['ar'].forward(enc_embeds.view(1, enc_embeds.size(0),
#                                                   enc_embeds.size(1)))
#
#     z_next = enc_embeds[1:]
#     c_t = c[0, :-1]
#     c_w, target_bin_pred = models_dict['pred'].forward(c_t)
#     z_preds = torch.matmul(c_w, z_next.t()).exp()
#
#     logits = F.log_softmax(z_preds, dim=1)
#     loss1 += logits.diag().mean()
#     loss2 += F.cross_entropy(target_bin_pred, target_bin[x_idx, :-1])
#
# loss = loss1 + loss2
#
# # calc metrics
# max_out = torch.argmax(target_bin_pred, 1)
# bin_centroids = (torch_bins[max_out] + torch_bins[max_out + 1]) / 2
# bin_centroids = bin_centroids.to(device=cuda, non_blocking=True)
# metrics = F.l1_loss(bin_centroids, target[x_idx, :-1])
#
# loss_val_batch[0].append(loss.item())
# loss_val_batch[1].append(loss1.item())
# loss_val_batch[2].append(loss2.item())
# metrics_val_batch.append(metrics.item())

