

from daskms import xds_from_table
import numpy as np
import tempfile
import os
import subprocess
import shutil

def test_skysim_flux_conservation():
    """ CI test for skysim flux conservation"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        original_dir = os.getcwd()
        os.chdir(tmpdir)
        
        try:
            # 1. Create catalogue
            catalog = """#format: name ra dec stokes_i emaj emin pa
SRC 00:00:00.0 -30.00.00.0 10.0 0.0 0.0 0.0
"""
            with open('test_catalog.txt', 'w') as f:
                f.write(catalog)
            
            # 2. Create empty MS 
            empty_ms_cmd = [
                'telsim', 'test_flux.ms',
                '-tel', 'meerkat',
                '-dir', 'J2000,0deg,-30deg',
                '-st', '2025-03-06T12:25:00',
                '-dt', '60',
                '-nt', '5',
                '-sf', '1420MHz',
                '-df', '1MHz', 
                '-nc', '1',
                '-corr', 'XX,YY',
                '-rc', '1000',
                '-col', 'DATA'
            ]
            
            result = subprocess.run(empty_ms_cmd, capture_output=True, text=True, timeout=60)
            print("telsim STDOUT:", result.stdout)
            print("telsim STDERR:", result.stderr)
            print("telsim Return code:", result.returncode)
            
            if result.returncode != 0:
                assert False, f"telsim failed to create empty MS: {result.stderr}"
            
            # 3. run SkySIM
            skysim_cmd = [
                'skysim',
                '--ms', 'test_flux.ms',
                '--catalogue', 'test_catalog.txt',
                '--column', 'DATA',
                '--cat-delim', ' ',
                '--mode', 'sim'
            ]
            
            result = subprocess.run(skysim_cmd, capture_output=True, text=True, timeout=120)
            
            print("skysim STDOUT:", result.stdout)
            print("skysim STDERR:", result.stderr)
            print("skysim Return code:", result.returncode)
            
            assert result.returncode == 0, f"skysim failed: {result.stderr}"
            
            # 4. Verify results
            ds = xds_from_table('test_flux.ms')[0]
            data = ds.DATA.values
            amplitude = np.abs(data)
            mean_amplitude = np.mean(amplitude)
            
            expected_flux = 10.0
            assert np.isclose(mean_amplitude, expected_flux, rtol=0.05), \
                f"Flux {mean_amplitude:.3f} not within 5% of expected {expected_flux}"
            
            print(f"✓ SUCCESS: Flux conserved! {mean_amplitude:.3f} Jy")
            
        finally:
            os.chdir(original_dir)
            # cleanup
            if os.path.exists(os.path.join(tmpdir, 'test_flux.ms')):
                shutil.rmtree(os.path.join(tmpdir, 'test_flux.ms'), ignore_errors=True)

if __name__ == "__main__":
    test_skysim_flux_conservation()

# alternatively, use the observe.yaml recipe 

    