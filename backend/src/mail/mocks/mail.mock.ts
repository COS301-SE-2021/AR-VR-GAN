import { MailerService } from '@nestjs-modules/mailer';
import { Injectable } from '@nestjs/common';
import { sendEmailDto } from '../dto/send-email.dto';

export const MockMailService = {
    sendConfirmationEmail: jest.fn((dto) => {
        return {
          ...dto
        }
      })
}



export default class MockMailClass {
  public sendConfirmationEmail(request: sendEmailDto) {
    return "test";
  }
  public testFunc(){

  }

}
