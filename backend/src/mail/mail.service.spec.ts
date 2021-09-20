import { Test, TestingModule } from '@nestjs/testing';
import { MailService } from './mail.service';
import { MailModule } from './mail.module';
import { sendEmailDto } from './dto/send-email.dto';
import MockMailClass, { MockMailService } from './mocks/mail.mock';

describe('MailService', () => {
  let service: MailService;
  let mockService: MockMailClass

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [MailService],
      imports:[
        MailModule
      ]
    }).overrideProvider(MailService).useValue(MockMailService).compile();

    service = module.get<MailService>(MailService);
     
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  it('send mail definied', async () => {
    const emailDto = new sendEmailDto("test","test","test");
    expect(await service.sendConfirmationEmail(emailDto)).toBeDefined();
  });


});
