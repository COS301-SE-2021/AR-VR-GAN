import { Controller, Post, UseInterceptors, UploadedFile, Res, Body } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { Response } from 'express';
import { diskStorage } from 'multer';
import { UploadService } from './upload.service';
import { RequestBody } from './interfaces/coordinates.interface';

@Controller('upload')
export class UploadController {
    constructor(private readonly uploadService: UploadService) {}

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
        const response = {
            original_filename: file.originalname,
            new_filename: file.filename,
        };

        return response;
    }

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