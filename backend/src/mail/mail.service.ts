import { MailerService } from '@nestjs-modules/mailer';
import { Injectable } from '@nestjs/common';
import { sendEmailDto } from './dto/send-email.dto';

@Injectable()
export class MailService {
  constructor(private mailerService: MailerService) {}

  async sendConfirmationEmail(request: sendEmailDto) {
    await this.mailerService.sendMail({
      to: request.email,
      // from: '"Support Team" <javacinsomniacs@gmail.com>',
      subject: 'Your Model is ready to use!',
      template: './doneTraining',
      context: {      
        name: request.username,
        modelName: request.modelName
      },
    });
  }
}
