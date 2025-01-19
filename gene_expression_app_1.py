import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
import joblib
import os
import seaborn as sns
import matplotlib.pyplot as plt

class GeneExpressionApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Gene Expression Analysis Tool")
        self.master.geometry("600x650")  # Adjusted for more space
        self.master.configure(background="#E0FFFF")  # Set background color to a lighter blue tone
        self.font_style = ("Calibri", 12)
        self.heading_font = ("Calibri", 20, "bold")  # Font for the heading

        self.file_path = None
        self.selected_features = None  # To store the selected features used during training

        # Create a frame for heading
        self.heading_frame = tk.Frame(self.master, bg="#E0FFFF")  # Lighter blue background for the heading frame
        self.heading_frame.pack(pady=20)

        # DNA Symbol (could be an image or emoji)
        self.dna_symbol = "🧬"  # Using an emoji as the DNA symbol
        self.heading_label = tk.Label(self.heading_frame, text=f"{self.dna_symbol} Gene Expression Analysis Tool", font=self.heading_font, fg="blue", bg="#E0FFFF")
        self.heading_label.pack()

        # Create a frame for buttons and progress bar
        self.button_frame = tk.Frame(self.master, bg="#e0e0e0", padx=10, pady=10)
        self.button_frame.pack(pady=10)

        # Use a grid layout for the buttons to ensure they are well aligned
        button_color = "#ADD8E6"  # Light blue color

        # Adding black border and relief to buttons
        self.load_data_button = tk.Button(self.button_frame, text="Load Data", command=self.load_data, width=15, bg=button_color, fg="black", font=self.font_style, borderwidth=2, relief="solid")
        self.load_data_button.grid(row=0, column=0, padx=5, pady=5)

        self.run_analysis_button = tk.Button(self.button_frame, text="Run Analysis", command=self.run_analysis, width=15, bg=button_color, fg="black", font=self.font_style, borderwidth=2, relief="solid")
        self.run_analysis_button.grid(row=0, column=1, padx=5, pady=5)

        self.train_model_button = tk.Button(self.button_frame, text="Train Models", command=self.train_models, width=15, bg=button_color, fg="black", font=self.font_style, borderwidth=2, relief="solid")
        self.train_model_button.grid(row=1, column=0, padx=5, pady=5)

        self.export_data_button = tk.Button(self.button_frame, text="Export Results", command=self.export_data, width=15, state=tk.DISABLED, bg=button_color, fg="black", font=self.font_style, borderwidth=2, relief="solid")
        self.export_data_button.grid(row=1, column=1, padx=5, pady=5)

        # Progress bar with color matching the heading
        self.progress_style = ttk.Style()
        self.progress_style.theme_use('default')
        self.progress_style.configure("Color.Horizontal.TProgressbar", foreground='blue', background='blue')
        self.progress_bar = ttk.Progressbar(self.master, orient="horizontal", length=300, mode="indeterminate", style="Color.Horizontal.TProgressbar")
        self.progress_bar.pack(pady=10)

        # Create a frame for text display
        self.text_frame = tk.Frame(self.master, bg="#e0e0e0", padx=10, pady=10)
        self.text_frame.pack(pady=10)

        self.results_text = tk.Text(self.text_frame, height=15, width=80, font=("Arial", 12), bg="#ffffee")
        self.results_text.pack(pady=5)

        self.results_text.insert(tk.END, "Results will be displayed here.")

        # Help Button at the top right corner
        self.help_button = tk.Button(self.master, text="Help", command=self.show_help, width=10, bg=button_color, fg="black", font=self.font_style, borderwidth=2, relief="solid")
        self.help_button.place(x=500, y=20)  # Place it at the top right corner

        # Confusion Matrix Button
        self.confusion_matrix_button = tk.Button(self.master, text="Show Confusion Matrix", command=self.show_confusion_matrix, width=20, bg="blue", fg="white", font=("Calibri", 12), borderwidth=2, relief="solid")
        self.confusion_matrix_button.pack(pady=10)

    def show_help(self):
        help_message = (
            "Gene Expression Analysis Tool\n\n"
            "This tool performs gene expression analysis on provided datasets.\n"
            "Steps to use:\n"
            "1. Load a CSV file containing gene expression data.\n"
            "2. Run analysis to predict the gene type.\n"
            "3. Train models to create classifiers and save them.\n"
            "4. Export the analysis results to a CSV file.\n\n"
            "For more information, contact the developer at: areebamansoor90@gmail.com"
        )
        messagebox.showinfo("Help - Gene Expression Analysis Tool", help_message)

    def load_data(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if self.file_path:
            messagebox.showinfo("Success", f"Data loaded successfully from: {self.file_path}")
            self.export_data_button.config(state=tk.NORMAL)  # Enable export button after data is loaded
        else:
            messagebox.showerror("Error", "No file selected.")

    def run_analysis(self):
        if self.file_path:
            try:
                self.progress_bar.start()  # Start the progress bar

                # Specify data types for columns (adjust based on your data)
                dtypes = {'samples': str, 'type': str}  # Adjust with actual column names

                df = pd.read_csv(self.file_path, dtype=dtypes)
                df_imputed = self.impute_missing_values(df)
                X_reduced, pca, classifier, label_encoder = self.load_model_and_transform_data(df_imputed)
                predictions = self.make_predictions(X_reduced, classifier)
                df['predicted_type'] = label_encoder.inverse_transform(predictions)
                self.display_results(df[['samples', 'type', 'predicted_type']])
            except Exception as e:
                messagebox.showerror("Error", str(e))
            finally:
                self.progress_bar.stop()  # Stop the progress bar
        else:
            messagebox.showerror("Error", "Please load data first.")

    def train_models(self):
        if self.file_path:
            try:
                self.progress_bar.start()

                df = pd.read_csv(self.file_path)
                features = df.select_dtypes(include=[np.number]).iloc[:, 2:]  # Select numeric columns starting from index 2
                target = df['type']

                # Store the selected features for later use
                self.selected_features = features.columns.tolist()

                # Preprocess data
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)

                # Train PCA
                pca = PCA(n_components=2)  # Adjust as necessary
                X_reduced = pca.fit_transform(features_scaled)

                # Encode target variable
                label_encoder = LabelEncoder()
                y_encoded = label_encoder.fit_transform(target)

                # Train classifier
                classifier = RandomForestClassifier(random_state=42)
                classifier.fit(X_reduced, y_encoded)

                # Save models
                joblib.dump(pca, "pca_model.pkl")
                joblib.dump(classifier, "classifier_model.pkl")
                joblib.dump(label_encoder, "label_encoder.pkl")

                messagebox.showinfo("Success", "Models trained and saved successfully!")
            except Exception as e:
                messagebox.showerror("Error", f"Model training failed: {str(e)}")
            finally:
                self.progress_bar.stop()
        else:
            messagebox.showerror("Error", "Please load a dataset first.")

    def impute_missing_values(self, df):
        # Example imputation: filling missing values with the column mean
        df_imputed = df.copy()
        for column in df_imputed.select_dtypes(include=[np.number]):  # For numeric columns
            df_imputed[column] = df_imputed[column].fillna(df_imputed[column].mean())  # Avoid inplace=True
        return df_imputed

    def load_model_and_transform_data(self, df):
        # Example: Load PCA model and classifier, assuming they are saved in joblib format
        try:
            # Check if the model files exist
            if not os.path.exists("pca_model.pkl") or not os.path.exists("classifier_model.pkl") or not os.path.exists("label_encoder.pkl"):
                raise FileNotFoundError("One or more model files are missing. Please train the models first.")

            # Load previously trained models
            pca = joblib.load("pca_model.pkl")
            classifier = joblib.load("classifier_model.pkl")
            label_encoder = joblib.load("label_encoder.pkl")

            # Ensure the dataframe has the same features as the training data
            features = df[self.selected_features]  # Use the same features as trained on
            print(f"Features for PCA: {features.columns.tolist()}")  # Debug: Check selected features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)

            # Apply PCA and ensure consistent columns
            print(f"Features shape before PCA: {features_scaled.shape}")  # Debug: Check shape before PCA
            X_reduced = pca.transform(features_scaled)
            print(f"X_reduced shape after PCA: {X_reduced.shape}")  # Debug: Check shape after PCA

            return X_reduced, pca, classifier, label_encoder
        except FileNotFoundError as e:
            raise ValueError(str(e))
        except Exception as e:
            raise ValueError("Error loading model or transforming data: " + str(e))

    def make_predictions(self, X_reduced, classifier):
        # Make predictions using the classifier
        predictions = classifier.predict(X_reduced)
        return predictions

    def display_results(self, df):
        # Display results in the Text widget
        self.results_text.delete(1.0, tk.END)  # Clear previous results
        self.results_text.insert(tk.END, df.to_string(index=False))  # Display new results

    def export_data(self):
        # Export results to a CSV file
        export_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if export_path:
            try:
                self.results_text.get("1.0", tk.END)  # Ensure data is up-to-date
                df = pd.read_csv(self.file_path)  # Or use self.df if stored
                df.to_csv(export_path, index=False)
                messagebox.showinfo("Success", f"Results exported to {export_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to export results: {str(e)}")

    def show_confusion_matrix(self):
        if self.file_path:
            try:
                df = pd.read_csv(self.file_path)
                features = df[self.selected_features]
                target = df['type']
                
                # Preprocess data
                scaler = StandardScaler()
                features_scaled = scaler.fit_transform(features)

                # Load the trained classifier and label encoder
                classifier = joblib.load("classifier_model.pkl")
                label_encoder = joblib.load("label_encoder.pkl")
                
                # Transform the data and make predictions
                X_reduced = PCA(n_components=2).fit_transform(features_scaled)
                predictions = classifier.predict(X_reduced)
                
                # Compute confusion matrix
                cm = confusion_matrix(target, label_encoder.inverse_transform(predictions))
                
                # Plot confusion matrix using Seaborn
                plt.figure(figsize=(8, 6))
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
                plt.title("Confusion Matrix", fontsize=16)
                plt.xlabel("Predicted", fontsize=12)
                plt.ylabel("Actual", fontsize=12)
                plt.show()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to show confusion matrix: {str(e)}")
        else:
            messagebox.showerror("Error", "Please load data and run analysis first.")


# Main application loop
if __name__ == "__main__":
    root = tk.Tk()
    app = GeneExpressionApp(root)
    root.mainloop()
