# FlashFilter

This is a proof of concept and a work in progress. It is not intended for production use.

## Description
- Using image subtraction to detect area of flash in a frame. (Assuming previous frame is corrected and has no flash)
- Generate a bitmask of the flash area.
- Use interpolation to fill the flash area with the surrounding pixels in spatial and temporal domain.

## Usage
Run the prep_data.py script to prepare the data into the frames directory.

Run main.py and the output will be saved in the output directory.

## TODO
- Optimize interpolation methods and corresponding parameters to reduce artifacts.
- Add more frames that are used for interpolation.
- Try generative inpainting techniques.


