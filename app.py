import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import cv2

class CancerClassifier(nn.Module):
    def __init__(self, pretrained=True):
        super(CancerClassifier, self).__init__()
        # Use pretrained ResNet18
        self.resnet = models.resnet18(pretrained=pretrained)
        # Replace the final fully connected layer
        num_ftrs = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 1)
        )
        
    def forward(self, x):
        x = self.resnet(x)
        return x

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.target_layer.register_forward_hook(self.forward_hook)
        self.target_layer.register_full_backward_hook(self.backward_hook)
    
    def forward_hook(self, module, input, output):
        self.activations = output.detach()
    
    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, input_image, target_class=None):
        # Forward pass
        model_output = self.model(input_image)
        
        if target_class is None:
            target_class = 0  # For binary classification, we're interested in the positive class
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass with the target class
        if len(model_output.shape) == 1:
            model_output[target_class].backward()
        else:
            model_output[0, target_class].backward()
        
        # Get weights
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)
        
        # Generate CAM
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)
        cam = F.relu(cam)  # Apply ReLU to focus on features that have a positive influence
        
        # Normalize CAM
        cam = F.interpolate(cam, size=input_image.shape[2:], mode='bilinear', align_corners=False)
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam.squeeze().cpu().numpy()

class CancerDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Histopathologic Cancer Detection with Grad-CAM")
        self.root.geometry("1000x800")
        self.root.configure(bg="#f0f0f0")
        
        # Set up the model
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.setup_model()
        
        # Set up transformation for images
        self.transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Create UI elements
        self.create_widgets()
        
        # Variables
        self.current_image_path = None
        self.original_image = None
        self.processed_tensor = None
        self.fig = None
        self.canvas = None
        
        # Set up Grad-CAM
        target_layer = self.model.resnet.layer4[-1].conv2
        self.grad_cam = GradCAM(self.model, target_layer)

    def setup_model(self):
        try:
            # Initialize model
            self.model = CancerClassifier(pretrained=False).to(self.device)
            
            # Try to load the trained model
            model_path = 'best_model.pth'
            if os.path.exists(model_path):
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                print(f"Model loaded from {model_path}")
            else:
                messagebox.showwarning("Model Not Found", f"Could not find model at {model_path}. The app will continue but predictions will be random.")
                print(f"Warning: Model not found at {model_path}")
            
            # Set to evaluation mode
            self.model.eval()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load model: {str(e)}")
            print(f"Error loading model: {str(e)}")
            sys.exit(1)

    def create_widgets(self):
        # Title Label
        title_label = tk.Label(self.root, text="Histopathologic Cancer Detection with Grad-CAM", 
                               font=("Arial", 18, "bold"), bg="#f0f0f0")
        title_label.pack(pady=20)
        
        # Instructions
        instructions = tk.Label(self.root, 
                                text="Select a histopathologic image to detect the presence of cancer cells. Grad-CAM will highlight regions the model focuses on.",
                                font=("Arial", 12), bg="#f0f0f0", wraplength=600)
        instructions.pack(pady=10)
        
        # Main content frame (for images and visualization)
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(pady=10, fill=tk.BOTH, expand=True)
        
        # Frame for the original image
        self.image_frame = tk.Frame(main_frame, bg="#ffffff", width=300, height=300, 
                                   relief=tk.SUNKEN, bd=2)
        self.image_frame.grid(row=0, column=0, padx=20, pady=10)
        self.image_frame.grid_propagate(False)  # Prevent frame from shrinking
        
        # Default image placeholder
        self.image_label = tk.Label(self.image_frame, bg="#ffffff", 
                                    text="No image selected", font=("Arial", 12))
        self.image_label.pack(fill=tk.BOTH, expand=True)
        
        # Title for original image
        orig_title = tk.Label(main_frame, text="Original Image", font=("Arial", 12, "bold"), bg="#f0f0f0")
        orig_title.grid(row=1, column=0, pady=5)
        
        # Frame for the Grad-CAM visualization
        self.gradcam_frame = tk.Frame(main_frame, bg="#ffffff", width=300, height=300, 
                                     relief=tk.SUNKEN, bd=2)
        self.gradcam_frame.grid(row=0, column=1, padx=20, pady=10)
        self.gradcam_frame.grid_propagate(False)  # Prevent frame from shrinking
        
        # Default Grad-CAM placeholder
        self.gradcam_label = tk.Label(self.gradcam_frame, bg="#ffffff", 
                                      text="Grad-CAM will appear here", font=("Arial", 12))
        self.gradcam_label.pack(fill=tk.BOTH, expand=True)
        
        # Title for Grad-CAM image
        gradcam_title = tk.Label(main_frame, text="Grad-CAM Visualization", font=("Arial", 12, "bold"), bg="#f0f0f0")
        gradcam_title.grid(row=1, column=1, pady=5)
        
        # Button Frame
        button_frame = tk.Frame(self.root, bg="#f0f0f0")
        button_frame.pack(pady=20)
        
        # Select Image Button
        select_btn = tk.Button(button_frame, text="Select Image", command=self.select_image,
                              font=("Arial", 12), bg="#4CAF50", fg="white",
                              padx=20, pady=10)
        select_btn.grid(row=0, column=0, padx=10)
        
        # Predict Button
        predict_btn = tk.Button(button_frame, text="Detect Cancer", command=self.predict,
                               font=("Arial", 12), bg="#2196F3", fg="white",
                               padx=20, pady=10)
        predict_btn.grid(row=0, column=1, padx=10)
        
        # Result Frame
        self.result_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.result_frame.pack(pady=10, fill=tk.X, padx=50)
        
        # Result Label
        self.result_label = tk.Label(self.result_frame, text="", font=("Arial", 14, "bold"),
                                   bg="#f0f0f0")
        self.result_label.pack()
        
        # Confidence Bar Frame
        self.conf_frame = tk.Frame(self.result_frame, bg="#f0f0f0", height=30)
        self.conf_frame.pack(fill=tk.X, pady=10)
        
        # Explanation Frame
        explanation_frame = tk.Frame(self.root, bg="#f0f0f0")
        explanation_frame.pack(pady=10, fill=tk.X, padx=50)
        
        # Explanation Label
        explanation_text = """
        Grad-CAM Explanation:
        The heatmap shows regions where the model is focusing to make its prediction.
        Red areas indicate regions strongly associated with cancer detection.
        Blue areas have less influence on the cancer prediction.
        """
        explanation_label = tk.Label(explanation_frame, text=explanation_text, 
                                     font=("Arial", 10), bg="#f0f0f0", 
                                     justify=tk.LEFT, wraplength=800)
        explanation_label.pack(anchor="w")
        
        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, 
                             relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)

    def clear_matplotlib_figure(self):
        """Properly clean up matplotlib figure and canvas"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
        
        if self.canvas is not None:
            self.canvas.get_tk_widget().destroy()
            self.canvas = None
            
        # Reset the gradcam frame contents
        for widget in self.gradcam_frame.winfo_children():
            widget.destroy()
            
        self.gradcam_label = tk.Label(self.gradcam_frame, bg="#ffffff", 
                                      text="Grad-CAM will appear here", font=("Arial", 12))
        self.gradcam_label.pack(fill=tk.BOTH, expand=True)

    def select_image(self):
        # Open file dialog
        filetypes = [
            ("Image files", "*.jpg *.jpeg *.png *.tif *.tiff *.bmp"),
            ("All files", "*.*")
        ]
        image_path = filedialog.askopenfilename(title="Select Image", filetypes=filetypes)
        
        if image_path:
            try:
                # Clean up previous matplotlib objects
                self.clear_matplotlib_figure()
                
                # Load and display the image
                self.current_image_path = image_path
                self.status_var.set(f"Selected image: {os.path.basename(image_path)}")
                
                # Clear previous results
                self.result_label.config(text="")
                for widget in self.conf_frame.winfo_children():
                    widget.destroy()
                
                # Load and save original image
                self.original_image = Image.open(image_path)
                if self.original_image.mode != 'RGB':
                    self.original_image = self.original_image.convert('RGB')
                
                # Resize image for display
                display_image = self.original_image.resize((280, 280), Image.LANCZOS)
                photo = ImageTk.PhotoImage(display_image)
                
                # Update the image label
                self.image_label.config(image=photo, text="")
                self.image_label.image = photo  # Keep a reference
                
                # Preprocess for model
                self.processed_tensor = self.transform(self.original_image).unsqueeze(0).to(self.device)
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {str(e)}")
                self.status_var.set("Error loading image")
                print(f"Image loading error: {str(e)}")
                import traceback
                traceback.print_exc()

    def predict(self):
        if not self.current_image_path or self.processed_tensor is None:
            messagebox.showinfo("No Image", "Please select an image first")
            return
        
        try:
            self.status_var.set("Analyzing image...")
            self.root.update()
            
            # Make prediction
            with torch.no_grad():
                output = self.model(self.processed_tensor)
                probability = torch.sigmoid(output).item()
            
            # Display result
            for widget in self.conf_frame.winfo_children():
                widget.destroy()
            
            # Create confidence bar
            cancer_prob = probability * 100
            non_cancer_prob = (1 - probability) * 100
            
            # Choose colors based on prediction
            if probability >= 0.5:
                result_text = f"Result: Cancer Detected ({cancer_prob:.1f}%)"
                result_color = "#e74c3c"  # Red
            else:
                result_text = f"Result: No Cancer Detected ({non_cancer_prob:.1f}%)"
                result_color = "#2ecc71"  # Green
            
            self.result_label.config(text=result_text, fg=result_color)
            
            # Create confidence bar
            bar_height = 20
            bar_width = self.conf_frame.winfo_width() if self.conf_frame.winfo_width() > 1 else 400
            
            # Create canvas for the confidence bar
            canvas = tk.Canvas(self.conf_frame, height=bar_height, width=bar_width, 
                               bg="#f0f0f0", highlightthickness=0)
            canvas.pack(fill=tk.X)
            
            # Draw the bar
            # Non-cancer portion (green)
            non_cancer_width = int(bar_width * (1 - probability))
            canvas.create_rectangle(0, 0, non_cancer_width, bar_height, 
                                   fill="#2ecc71", outline="")
            
            # Cancer portion (red)
            canvas.create_rectangle(non_cancer_width, 0, bar_width, bar_height, 
                                   fill="#e74c3c", outline="")
            
            # Add labels
            canvas.create_text(bar_width * 0.25, bar_height // 2, 
                              text=f"Non-Cancer: {non_cancer_prob:.1f}%", 
                              fill="black", font=("Arial", 9, "bold"))
            
            canvas.create_text(bar_width * 0.75, bar_height // 2, 
                              text=f"Cancer: {cancer_prob:.1f}%", 
                              fill="black", font=("Arial", 9, "bold"))
            
            # Generate Grad-CAM
            self.generate_gradcam()
            
            self.status_var.set("Analysis complete")
            
        except Exception as e:
            messagebox.showerror("Error", f"Error during prediction: {str(e)}")
            self.status_var.set("Error during analysis")
            print(f"Prediction error: {str(e)}")
            import traceback
            traceback.print_exc()

    def generate_gradcam(self):
        """Generate and display Grad-CAM visualization"""
        try:
            # Clean up previous matplotlib objects
            self.clear_matplotlib_figure()
            
            # Generate class activation map
            cam = self.grad_cam.generate_cam(self.processed_tensor)
            
            # Convert the original image to numpy array for visualization
            img_array = np.array(self.original_image.resize((96, 96)))
            
            # Create heatmap
            heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            
            # Superimpose heatmap on original image
            superimposed = cv2.addWeighted(img_array, 0.6, heatmap, 0.4, 0)
            
            # Create figure for plotting
            self.fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4))
            
            # Original with heatmap overlay
            ax1.imshow(superimposed)
            ax1.set_title('Combined')
            ax1.axis('off')
            
            # Just the heatmap
            ax2.imshow(heatmap)
            ax2.set_title('Heatmap')
            ax2.axis('off')
            
            plt.tight_layout()
            
            # Embed the matplotlib figure in tkinter
            for widget in self.gradcam_frame.winfo_children():
                widget.destroy()
                
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.gradcam_frame)
            self.canvas.draw()
            self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
            
        except Exception as e:
            messagebox.showerror("Error", f"Error generating Grad-CAM: {str(e)}")
            self.status_var.set("Error generating Grad-CAM")
            print(f"Grad-CAM error: {str(e)}")
            import traceback
            traceback.print_exc()

    def save_results(self):
        """Save the predictions and visualization"""
        if not self.current_image_path:
            messagebox.showinfo("No Results", "No results to save")
            return
            
        save_path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
            title="Save Results As"
        )
        
        if save_path:
            # TODO: Implement saving functionality
            pass

def main():
    root = tk.Tk()
    app = CancerDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()