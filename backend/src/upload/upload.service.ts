import { Injectable } from '@nestjs/common';
import { extname } from 'path';
import { Request } from '../model/interfaces/request.interface';

@Injectable()
export class UploadService {
    static customeFileName(req, file, callback) {
        const name = file.originalname.split('.')[0];
        
        const fileExtName = extname(file.originalname);
    
        const randomName = Array(4)
            .fill(null)
            .map(() => Math.round(Math.random() * 16).toString(16))
            .join('');
    
        callback(null, `${name}-${randomName}${fileExtName}`);
    }

    public handleCoords(request: Request): number {
        let sum = 0;

        for (let i = 0; i < request.data.length; i++) {
            sum += request.data[i]
        }

        return sum;
    }
}
