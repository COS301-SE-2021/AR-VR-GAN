import { Controller, Post, UseInterceptors, UploadedFile, Res, Body, Get } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { Response } from 'express';
import { diskStorage } from 'multer';
import { UploadService } from './upload.service';
import { RequestBody } from './interfaces/coordinates.interface';

@Controller('upload')
export class UploadController {
    constructor(private readonly uploadService: UploadService) {}

    /**
     * handles the post request to upload a file to the server
     * @param file file to be saved
     * @returns 
     */
    @Post('file')
    @UseInterceptors(
        FileInterceptor('file', {
            dest: './uploads',
            storage: diskStorage({
                destination: './uploads',
                filename: UploadService.customeFileName
            })
        })
    )
    async uploadedFile(@UploadedFile() file: Express.Multer.File) {
        console.log("here")
        const response = {
            original_filename: file.originalname,
            new_filename: file.filename,
        };

        return response;
    }

    /**
     * handles the post request to retrieve an image from given coordinates
     * @param res 
     * @param requestBody 
     */
    @Post('getImageFromCoordinates')
    getImageFromCoordinates(@Res() res: Response, @Body() requestBody: RequestBody) {
        let sum = Math.floor(this.uploadService.handleCoords(requestBody)) % 10;
        let filename = sum.toString() + '.jpg';

        const options = {
            root: './uploads',
            dotfiles: 'deny',
            headers: {
              'x-timestamp': Date.now(),
              'x-sent': true
            }
        }
    
        res.sendFile(filename, options)
    }

}