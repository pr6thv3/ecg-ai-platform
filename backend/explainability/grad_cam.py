import torch
import torch.nn.functional as F
import numpy as np

class GradCAM1D:
    def __init__(self, model, target_layer_name=None):
        self.model = model
        self.model.eval()
        self.target_layer_name = target_layer_name
        self.gradients = None
        self.activations = None
        
        # Hook the target layer
        # If target_layer_name is not provided, try to find the last conv layer
        target_layer = self._get_target_layer()
        if target_layer is not None:
            target_layer.register_forward_hook(self.save_activation)
            target_layer.register_full_backward_hook(self.save_gradient)

    def _get_target_layer(self):
        if self.target_layer_name:
            for name, module in self.model.named_modules():
                if name == self.target_layer_name:
                    return module
            return None
        else:
            # Default to the last Conv1d in features
            last_conv = None
            for module in self.model.modules():
                if isinstance(module, torch.nn.Conv1d):
                    last_conv = module
            return last_conv

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, target_class_idx):
        """
        Generate Grad-CAM saliency map for 1D input.
        Args:
            input_tensor: shape (1, 1, seq_len) or (1, seq_len)
            target_class_idx: int, target class index
        """
        if len(input_tensor.shape) == 2:
            input_tensor = input_tensor.unsqueeze(1)
            
        input_tensor.requires_grad = True
        
        # Forward pass
        output = self.model(input_tensor)
        
        # Get top predictions
        probs = F.softmax(output, dim=1).detach().cpu().numpy()[0]
        top_indices = np.argsort(probs)[::-1][:3]
        top_preds = [{"class": int(idx), "confidence": float(probs[idx])} for idx in top_indices]
        
        # Backward pass
        self.model.zero_grad()
        target = output[0, target_class_idx]
        target.backward(retain_graph=True)
        
        # Calculate Grad-CAM
        # gradients shape: (1, channels, seq_len')
        # activations shape: (1, channels, seq_len')
        weights = torch.mean(self.gradients, dim=2, keepdim=True) # Global average pooling over time
        cam = torch.sum(weights * self.activations, dim=1).squeeze(0) # Weighted sum of activations
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Interpolate to original sequence length
        cam = cam.unsqueeze(0).unsqueeze(0) # (1, 1, seq_len')
        cam = F.interpolate(cam, size=input_tensor.shape[2], mode='linear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        
        # Normalize between 0 and 1
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)
            
        return cam.tolist(), top_preds
