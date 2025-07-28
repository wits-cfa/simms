from astropy.io import fits
import dask.array as da
import xarray as xr
import numpy as np
from astropy.wcs import WCS
from scabha.basetypes import File
from astropy.table import Table
from astropy.coordinates import SpectralCoord
from astropy import units


class FitsData:
    def __init__(self, fname: str, memap: bool = True):
        self.fname = File(fname)
        if not self.fname.EXISTS:
            raise FileNotFoundError(f"Input FITS file '{fname}' does not exist")
            
        self.hdulist = fits.open(self.fname, memmap=memap)
        self.phdu = self.hdulist[0]
        self.header = self.phdu.header
        self.wcs = WCS(self.header)
        self.coord_names = self.wcs.axis_type_names[::-1]
        self.ndim = len(self.coord_names)
        self.dim_info = self.wcs.get_axis_types()[::-1]
        self.dshape = self.wcs.array_shape
        self.coords = xr.Coordinates()
        self.open_arrays = []
        self.data = da.asarray(self.phdu.data)

    @property
    def data(self):
        return self._data 
    
    @data.setter
    def data(self, value):
        self._data = value
    
    def set_coord_attrs(self, name:str):
        """ 
        Add FITS pixel meta data to this instances coords attribute (xarray.Coordinates)

        Args:
            name (str): Name (or label) of coordinate to set
        """
        
        idx = self.coord_names.index(name)
        self.coords[name].attrs = {
            "name": name,
            "pixel_size": self.pixel_scales[name],
            "dim": self.coords[name].dims[0],
            "ref_pixel": self.header[f"CRPIX{idx+1}"],
            "units": self.wcs.world_axis_units[::-1][idx],
            "size": self.dshape[idx]
        }
    
    def register_dimensions(self, set_dims=["spectral", "celestial"]):
        """
        Register (or set) FITS data coordinate and dimension data from the FITS header
        (including WCS information)
        
        Args:
            set_dims (List[str]): FITS coordinates (celestial, spectral, etc.) that must be set according to header WCS information. For example, if 'celestial' is part of this list, then the RA and DEC coordinate grid will be populated using the header WCS instead a dummy array. Avoid inlcuding dimesions for which you will need the coordinate grids for.
        """

        self.pixel_scales = {}
        pixel_scales = self.wcs.pixel_scale_matrix.diagonal()[::-1]
        
        
        for idx,coord in enumerate(self.coord_names):
            self.pixel_scales[coord] = pixel_scales[idx]
            diminfo = self.dim_info[idx]
            dim = diminfo["coordinate_type"]
            dtype = diminfo["group"]
            if dim == "celestial":
                continue
            elif dim == "spectral":    
                self.set_spectral_dimension(idx, dim not in set_dims)
                continue
                
            dimsize = self.dshape[idx]
            if dim in set_dims:
                dimgrid = getattr(self.wcs,
                                dim).array_index_to_world_values(da.arange(dimsize))
            else:
                dimgrid = da.empty(dimsize, dtype=dtype)
            self.coords[coord] = (dim,), dimgrid
            
        self.set_celestial_dimensions(empty = "celestial" not in set_dims)
        # re-traverse to set FITS pixel meta data
        for coord in self.coord_names:
            self.set_coord_attrs(coord)
            
        self.dims = list(self.coords.dims)
        
    def get_freq_from_vrad(self, rest_freq_Hz=None):
        """
        Convert radio velocity coordinates to frequencies

        Returns:
            astropy.SpectralCoord: Astropy SpectralCoord instance
        """
        
        rest_freq_Hz = rest_freq_Hz or self.header["RESTFRQ"]
        return SpectralCoord(self.coords["VRAD"], 
                unit=units.meter/units.second).to(units.Hz, 
                                            doppler_rest = rest_freq_Hz*units.Hz,
                                            doppler_convention="radio").value
        
    def get_freq_from_vopt(self):
        """
        Convert radio velocity coordinates to frequencies

        Returns:
            astropy.SpectralCoord: Astropy SpectralCoord instance
        """
        
        return SpectralCoord(self.coords["VOPT"],
                unit=units.meter/units.second).to(units.Hz, 
                                            doppler_rest = self.header["RESTFRQ"]*units.Hz,
                                            doppler_convention="optical").value

    def register_beam_info(self):
        """
        Get FITS beam information and assign it to a self.beam_info attribute:
        self.beam_info = {
            "bmaj": <array>,
            "bmin": <array>,
            "bpa": <array>, 
        }
        """
        header = self.header
        beam_table = None
        
        try:
            beam_table = Table.read(self.fname)
        except ValueError:
            pass
        
        beam_info = {}
        beam_info["bmaj"] = np.zeros(self.nchan)
        beam_info["bmin"] = np.zeros(self.nchan)
        beam_info["bpa"] = np.zeros(self.nchan)
    
        if beam_table:
            bunit = beam_table["BMAJ"].unit
            for chan in range(self.nchan):
                beam = beam_table[chan]
                beam_info["bmaj"][chan] = beam["BMAJ"]
                beam_info["bmin"][chan] = beam["BMIN"]
                beam_info["bpa"][chan] = beam["BPA"]
                
        elif header.get(f"BMAJ1", False):
            bunit = self.coords["RA"].attrs["units"]
            for chan in range(self.nchan):
                beam_info["bmaj"][chan] = header[f"BMAJ{chan+1}"]
                beam_info["bmin"][chan] = header[f"BMIN{chan+1}"]
                beam_info["bpa"][chan] = header[f"BPA{chan+1}"]
        elif header.get("BMAJ", False):
            bunit = self.coords["RA"].attrs["units"]
            if self.spectral_coord == "VRAD":
                freqs = self.get_freq_from_vrad()
            elif self.spectral_coord == "VOPT":
                freqs = self.get_freq_from_vopt()
            else:
                freqs = self.coords["FREQ"].data
            
            for chan in range(self.nchan):
                scale_factor = freqs[self.spectral_refpix]/freqs[chan]
                beam_info["bmaj"][chan] = header[f"BMAJ"] * scale_factor
                beam_info["bmin"][chan] = header[f"BMIN"] * scale_factor
                beam_info["bpa"][chan] = header[f"BPA"]
        else:
            #TODO(mika): Print warning
            return
        
        # convert to radians
        if isinstance(bunit, str):
            bunit = getattr(units, bunit)
        
        beam_info["bmaj"] = (beam_info["bmaj"]*bunit).to(units.rad).value
        beam_info["bmin"] = (beam_info["bmin"]*bunit).to(units.rad).value
        beam_info["bpa"] = (beam_info["bpa"]*bunit).to(units.rad).value
        
        self.beam_info = beam_info
        return 0 
    
    def add_coord(self, name, dim, dimgrid):
        self.coords[name] = dim, dimgrid
        

    def set_spectral_dimension(self, idx:int, empty:bool=False):
        """
        set spectral coordinate data using the FITS WCS infomation

        Args:
            idx (int): Index of spectral dimension. 
            empty (bool, optiona): Set the coordinate grid as an empty array. Defaults to False.
        """
        dimsize = self.dshape[idx]
        
        coord = self.coord_names[idx]
        if empty:
            dimgrid = da.zeros(dimsize, dtype=self.dim_info[idx]["group"])
        else:
            dimgrid = self.wcs.spectral.array_index_to_world_values(da.arange(dimsize))                
        self.coords[coord] = ("spectral",), dimgrid
        self.nchan = dimsize
        self.spectral_coord = coord
        self.spectral_units = self.wcs.spectral.world_axis_units[0]
        self.spectral_refpix = self.header.get(f"CRPIX{self.ndim - idx}")
        
    def set_celestial_dimensions(self, empty:bool=True):
        """
        set celestial coordinates data using the FITS WCS infomation

        Args:
            empty (bool, optiona): Set the coordinate grids as an empty array. Defaults to True.
        """
        
        for idx, diminfo in enumerate(self.dim_info):
            dim = diminfo["coordinate_type"]
            dim_number = diminfo["number"]
            if dim == "celestial":
                if dim_number == 0:
                    ra_idx = idx
                elif dim_number == 1:
                    dec_idx = idx
                else:
                    raise ValueError(f"Unkown celestial dimension in WCS: {dim_number}")
                    
        ra_dim = self.coord_names[ra_idx]
        dec_dim = self.coord_names[dec_idx]
        ra_dimsize = self.dshape[ra_idx]
        dec_dimsize = self.dshape[dec_idx]
        dtype = self.dim_info[ra_idx]["group"]
            
        if empty:
            self.coords[dec_dim] = ("celestial.dec",), da.empty(dec_dimsize, dtype=dtype)
            self.coords[ra_dim] = ("celestial.ra",), da.empty(ra_dimsize, dtype=dtype)
            return

        grid = self.wcs.celestial.array_index_to_world_values(da.arange(ra_dimsize),
                                        da.arange(dec_dimsize))
        
        self.coords[dec_dim] = ("celesstial.ra",), grid[1]
        self.coords[ra_dim] = ("celestial.dec",), grid[0]

    def getdata(self, data_slice=None) -> np.ndarray:
        if data_slice:
            data = self.phdu.section[tuple(data_slice)]
        else:
            data = self.phdu.data
        self.open_arrays.append(data)
        
        return self.open_arrays[-1]
        
    def get_xds(self, data_slice=[],
                transpose=[], chunks=dict(RA=64,DEC=64), **kwargs):
        
        chunkit = {}
        for key,val in chunks.items():
            if key in self.coord_names:
                key = self.coords[key].attrs["dim"]
            chunkit[key] = val
            
        
        if len(data_slice) == 0:
            data_slice = [slice(None)]*self.ndim
        if len(transpose) == 0:
            transpose = self.dims
        data = da.asarray(self.phdu.section[tuple(data_slice)])
        
        return xr.DataArray(data,
            coords = self.coords,
            attrs = {
                "header": dict(self.header.items()),
            },
            **kwargs,
        ).chunk(chunkit)
        
    def __close__(self):
        self.hdulist.close()
        for data in self.open_arrays:
            del data
    
    def  close(self):
        self.__close__()
            
    def __exit__(self):
        self.__close__()
