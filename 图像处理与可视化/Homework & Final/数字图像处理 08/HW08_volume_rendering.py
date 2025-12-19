import nibabel as nib
import vtk
import numpy as np

option = 1

# Load the NIfTI image and extract data
if option == 1:
    image = nib.load('./nii_files/image_lr.nii')
    opacity = [50, 400]
    color = [0, 400]
    gradient = [150, 600]
    # shape: (124, 124, 73)
    # spacing: (1.5, 1.5, 1.5) (mm)
elif option == 2:
    image = nib.load('./nii_files/segmented_brain.nii')
    opacity = [500, 1000]
    color = [0, 1000]
    gradient = [500, 1000]
    # shape: (256, 256, 10)
    # spacing: (1, 1, 1.25) (mm)
elif option == 3:
    image = nib.load('./nii_files/OAS1_0001_mpr-1.nii')
    opacity = [900, 1000]
    color = [0, 1000]
    gradient = [800, 1000]
    # shape: (256, 256, 128)
    # spacing: (1, 1, 1.25) (mm)

# Get image dimensions and spacing
data = np.array(image.get_fdata(), dtype=np.float64)
dims = image.shape
spacing = tuple(image.header['pixdim'][1:4])

# Create vtkImageData object and set its properties
vtk_image = vtk.vtkImageData()
vtk_image.SetDimensions(dims)
vtk_image.SetSpacing(spacing)
vtk_image.SetOrigin(0, 0, 0)
vtk_image.AllocateScalars(vtk.VTK_DOUBLE, 1)

# Flatten the data to match VTK's expected input order (z, y, x)
flat_data = np.array(data, dtype=np.float64).transpose(2, 1, 0).ravel()

# Set the scalar values for the vtkImageData
vtk_image.GetPointData().GetScalars().SetVoidArray(flat_data, flat_data.size, 1)

# Set up transfer functions (opacity, color, gradient opacity)
opacity_transfer_function = vtk.vtkPiecewiseFunction()
opacity_transfer_function.AddPoint(0, 0.0)
opacity_transfer_function.AddSegment(opacity[0], 0.3, opacity[1], 0.5)
opacity_transfer_function.ClampingOff()

color_transfer_function = vtk.vtkColorTransferFunction()
color_transfer_function.AddRGBSegment(0, 0.0, 0.0, 0.0, 20, 0.2, 0.2, 0.2)
color_transfer_function.AddRGBSegment(color[0], 0.1, 0.1, 0, color[1], 1, 1, 0)

gradient_transfer_function = vtk.vtkPiecewiseFunction()
gradient_transfer_function.AddPoint(0, 0.0)
gradient_transfer_function.AddSegment(gradient[0], 0.1, gradient[1], 0.3)

# Set volume properties (how the data will be visualized)
volume_property = vtk.vtkVolumeProperty()
volume_property.SetScalarOpacity(opacity_transfer_function)
volume_property.SetColor(color_transfer_function)
volume_property.SetGradientOpacity(gradient_transfer_function)
volume_property.ShadeOn()
volume_property.SetInterpolationTypeToLinear()
volume_property.SetAmbient(1)
volume_property.SetDiffuse(0.9)
volume_property.SetSpecular(0.8)
volume_property.SetSpecularPower(10)

# Set up volume mapper and ray casting
volume_mapper = vtk.vtkFixedPointVolumeRayCastMapper()
volume_mapper.SetInputData(vtk_image)
volume_mapper.SetImageSampleDistance(5.0)

# Create a volume and apply the mapper and properties
volume = vtk.vtkVolume()
volume.SetMapper(volume_mapper)
volume.SetProperty(volume_property)

# Set up the renderer, render window, and interactor
renderer = vtk.vtkRenderer()
renderer.SetBackground(1, 1, 1)  # White background
renderer.AddVolume(volume)

render_window = vtk.vtkRenderWindow()
render_window.AddRenderer(renderer)
render_window.SetSize(750, 750)

interactor = vtk.vtkRenderWindowInteractor()
interactor.SetRenderWindow(render_window)

# Add light to the scene
light = vtk.vtkLight()
light.SetColor(0, 1, 1)
renderer.AddLight(light)

# Render the scene
render_window.Render()
interactor.Initialize()
interactor.Start()
