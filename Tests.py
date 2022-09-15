def test_sampling_correctness(x, best_latents, mask_noise, least_losses, batch_start, batch_stop, model, args):
    """Quick test to verify cIMLE correctness. The two printed
    tensors should be equal up to machine precision.
    """
    x_test = x[:4]
    l_test = best_latents[batch_start:batch_stop][:4]
    m_test = mask_noise[batch_start:batch_stop][:4]
    z_test =  {"mask_noise": m_test, "latents": l_test}
    loss_test = least_losses[batch_start:batch_stop][:4]

    losses, _, _ = model(x_test, z_test,
        mask_ratio=args.mask_ratio,
        reduction="batch")

    tqdm.write(f"{losses}  {loss_test}")
