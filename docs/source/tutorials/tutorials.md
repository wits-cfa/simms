
# skysim: Understanding the Complete Process - in a nutshell

[a draft]
[Might need to change the format to .sst?]

## Table of Contents

- [Step 1: Starting Point - Reading the Measurement Set](#step-1-reading-the-measurement-set)
- [Step 2: Sky Model Input](#step-2-sky-model-input)
- [Step 3: Creating Source Objects](#step-3-creating-source-objects)
- [Step 4: Direction Cosine Transformation](#step-4-direction-cosine-transformation)
- [Step 5: Visibility Computation](#step-5-visibility-computation)
- [Step 6: Final Write](#step-6-final-write)
- [Data Flow Summary](#data-flow-summary)
- [Additional Features](#additional-features)
- [DFT vs FFT](#dft-vs-fft)
- [Best Practices](#best-practices)
- [Command-Line Abbreviations](#command-line-abbreviations)

## Step 1: Reading the Measurement Set

Everything begins with reading the MS: granted the MS is created using `telsim` or you have an existing MS of your own: `skysim` takes your MS and uses `daskms` to read the ms tables and subtables. 

- The phase center: The RA/DEC of the observations pointing centre
- Channel frequencies: The frequency setup of the MS
- Number of rows, channels, and correlations: Structure of the MS
- Time and antenna information
- UVW coordinates: Telescope baseline positions

## Step 2: Sky Model Input

`skysim` now needs to know what to simulate. You can provide this in two ways:

### Option A: Source Catalogue 

1. **Read the catalogue**: which entails:
   - Parsing the format header
   - Reading source information row by row
   - Mapping columns based on format (PyBDSF, Aegean, WSClean, etc.)
   - **Note**: If your catalogue is not from [PyBDSF, Aegean, or WSClean], you must provide a custom mapping file.

2. **Create Source Dictionary**:

   ```python
   data = {
       'name': [...],
       'ra': [...],
       'dec': [...],
       'stokes_i': [...],
       # Additional parameters if present
       'emaj': [...],
       'emin': [...],
       'pa': [...],
   }
   ```

### Option B: FITS File

1. **Read the FITS**:
   - Load image data
   - Get WCS information
   - Extract pixel grid and values

2. **Transform to source format**:
   - Convert pixel positions to sky coordinates
   - Create intensity grids
   - Handle spectral and polarization information

## Step 3: Creating Source Objects
For each source, skysim creates:
[talk about what source means?]
1. **Basic Source Object**:
   ```python
   source = Source(
       name = source_name,
       ra = ra_value,  # Converted to radians
       dec = dec_value,  # Converted to radians
       emaj = major_axis,  # If extended
       emin = minor_axis,  # If extended
       pa = position_angle  # If extended
   )
   ```

2. **Spectrum Object**:
   ```python
   spectrum = Spectrum(
       stokes_i = intensity,
       stokes_q = q_pol,  # If polarized
       stokes_u = u_pol,  # If polarized
       stokes_v = v_pol,  # If polarized
       line_peak = peak,  # If spectral line
       line_width = width,  # If spectral line
       # ... 
   )
   ```
## Step 4: Direction Cosine Transformation
For each source:
```python
dra = source.ra - phase_center_ra
l = np.cos(dec) * np.sin(dra)
m = np.sin(dec) * np.cos(dec0) - np.cos(dec) * np.sin(dec0) * np.cos(dra)
```

This converts source positions to the coordinate system interferometers measure in.


## Step 5: Visibility Computation
skysim combines all this information to compute visibilities. 

Steps:

1. **Scale UVW coordinates**:
   ```python
   wavelengths = C / frequencies
   uvw_scaled = uvw / wavelengths
   ```

2. **For Each Source**:
   - Calculate phase terms
   - Apply source brightness
   - Handle extended structure if present
   - Handle polarization if source is polarized

3. **Combine All Sources**:
   - Add contributions from each source
   - Handle correlations (XX, YY, etc.)
   - Apply noise if requested

## Step 6: Final Write
skysim writes the computed visibilities back to your MS:
- Uses daskms for efficient writing
- Writes to your specified column
- Add/subtracts if requested

## Data Flow Summary:
```
MS Reading (daskms)
      ↓
Sky Model Input (Catalogue/FITS)
      ↓
Source Objects Creation
      ↓
Coordinate Transformation
      ↓
Visibility Computation
      ↓
Write to MS (daskms)
```

This process transforms your sky model (whether from catalogue or fits) into the visibilities a radio telescope would measure, taking into account:
- Telescope configuration (from MS)
- Source properties (from your input)
- Observing parameters (frequencies, times)
- Additional parameters (noise, polarization)

## Additional Features
When working with MS files containing multiple fields or spectral windows, you can specify which ones to use:

- Field ID (```--field-id```): select which field
- Spectral Window Id (--spwid): select which spw to use

```skysim -ms smallvis.ms --cat skymodel.txt --colum SIMULATED --field-id 1 --spwid 0```

2. DFT vs FFT: For computational effieciency

When provided a fits for the simulation, skysim automatically will choose between Direct Fourier Transform (DFT) and Fast Fourier Transform (FFT) based on image sparsity.
- DFT: used for sparse images where > 80% of pixels are below the brightness threshold
- FFT: Used for dense images

## Best Practices
# To Do:
- Provide separate fits files for each Stokes when simulating polarized sources
- Use custom mapping files for non-standard catalogue formats
- Ensure frequency ranges match between your fits images and MS when using fits as model input

# Not to Do's: 
- Ensure all required colums are provided for a basic simulation
- Avoid mismatched coordinates between your sky model and MS phase centre

# Command-Line Abbreviations: 
Refer to simms.cabs for a full list of abbreviations.

--cat : --catalogue         (source catalog file)
-fs   : --fits-sky          (FITS file input)
--col : --column            (output column)
--sc  : --simulated-column  (simulated data column)
--ic  : --input-column      (input column for add/subtract modes)
