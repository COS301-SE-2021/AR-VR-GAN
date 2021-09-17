import { MailerService } from '@nestjs-modules/mailer';
import { Injectable } from '@nestjs/common';
import { sendEmailDto } from './dto/send-email.dto';

@Injectable()
export class MailService {
  constructor(private mailerService: MailerService) {}

  /**
   * sends an email to the user with the doneTraining template
   * @param request the user details
   */
  async sendConfirmationEmail(request: sendEmailDto) {
    await this.mailerService.sendMail({
      to: request.email,
      subject: 'Your Model is ready to use!',
      template: './doneTraining',
      context: {      
        name: request.username,
        modelName: request.modelName
      },
    });
  }
}
