import { Controller, Post, UseInterceptors, UploadedFile } from '@nestjs/common';
import { FileInterceptor } from '@nestjs/platform-express';
import { diskStorage } from 'multer';
import { UploadService } from './upload.service';

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
}