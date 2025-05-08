# Séance du 20/03

Il faut ramener les valeurs dans la plage [0,1] avant d'envoyer dnas le vgg.

mean = torch.as_tensor(mean, dtype=tensor.dtype,device=tensor.device).view(-1, 1, 1)
