import nibabel as nib
import vtk
import numpy as np

option = 3
smoothing_enabled = True # Set this to False to disable smoothing

# Load the NIfTI image and extract data
if option == 1:
    image = nib.load('./nii_files/image_lr.nii')
    iso_value = 170
    smoothing_iterations = 2000
    opacity = 0.95
    # shape: (124, 124, 73)
    # spacing: (1.5, 1.5, 1.5) (mm)
elif option == 2:
    image = nib.load('./nii_files/segmented_brain.nii')
    iso_value = 500
    smoothing_iterations = 1000
    opacity = 0.95
    # shape: (256, 256, 10)
    # spacing: (1, 1, 1.25) (mm)
elif option == 3:
    image = nib.load('./nii_files/OAS1_0001_mpr-1.nii')
    iso_value = 450
    smoothing_iterations = 1000
    opacity = 1
    # shape: (256, 256, 128)
    # spacing: (1, 1, 1.25) (mm)

data = np.array(image.get_fdata(), dtype=np.float64)
print("Min value:", np.min(data))
print("Max value:", np.max(data))
dims = image.shape
spacing = tuple(image.header['pixdim'][1:4])

# Create vtkImageData object and set its properties
image = vtk.vtkImageData()
image.SetDimensions(dims)
image.SetSpacing(spacing)
image.SetOrigin(0, 0, 0)
image.AllocateScalars(vtk.VTK_DOUBLE, 1)  # Allocate scalar data (1 component)

# Convert the data to a 1D array that matches the correct order
flat_data = np.array(data, dtype=np.float64).transpose(2, 1, 0).ravel()  # Transpose to (z, y, x) order and flatten

# Set the scalar values for the vtkImageData
image.GetPointData().GetScalars().SetVoidArray(flat_data, flat_data.size, 1)

# Marching cubes to extract the isosurface
extractor = vtk.vtkMarchingCubes()
extractor.SetInputData(image)
extractor.SetValue(0, iso_value)  # Set the contour value for the isosurface

# Create a stripper to connect triangles into strips
stripper = vtk.vtkStripper()
stripper.SetInputConnection(extractor.GetOutputPort())

# Map the polydata to an actor
mapper = vtk.vtkPolyDataMapper()

# Optional smoothing step
if smoothing_enabled:
    smoother = vtk.vtkSmoothPolyDataFilter()
    smoother.SetInputConnection(stripper.GetOutputPort())
    smoother.SetNumberOfIterations(smoothing_iterations)  # Set number of iterations for smoothing
    smoother.Update()
    poly_data = smoother.GetOutput()  # Get the smoothed polydata
    mapper.SetInputData(poly_data)
else:
    mapper.SetInputConnection(stripper.GetOutputPort())  # Without smoothing

actor = vtk.vtkActor()
actor.SetMapper(mapper)
actor.GetProperty().SetColor(1, 1, 0.1)  # yellow
actor.GetProperty().SetOpacity(opacity)  # Set opacity
actor.GetProperty().SetAmbient(0.05)  # Set ambient lighting
actor.GetProperty().SetDiffuse(0.5)  # Set diffuse lighting
actor.GetProperty().SetSpecular(1.0)  # Set specular lighting

# Set up the rendering window and interactor
renderer = vtk.vtkRenderer()
renderer.SetBackground(1, 1, 1)  # White background
renderer.AddActor(actor)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(750, 750)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)
interactor.Initialize()

# Start rendering
render_window.Render()
interactor.Start()


