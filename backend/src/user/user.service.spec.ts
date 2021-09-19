import { Test, TestingModule } from '@nestjs/testing';
import { UserService } from './user.service';
import { MockUserModel } from './mocks/userRepository.mock'
import { getModelToken } from '@nestjs/mongoose';
import { JwtService } from '@nestjs/jwt';
import { LoginUserDto } from './dto/login-user.dto';
import { UserResponse } from './dto/user-response.dto';

describe('UserService', () => {
  let service: UserService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [ 
        UserService,
        {
          provide : getModelToken('User'),
          useClass : MockUserModel
        },
        {
          provide: JwtService,
          useValue: JwtService
        }],
    }).compile();

    service = module.get<UserService>(UserService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  it('login-false : user not found', async () => {
    const loginDto = new LoginUserDto("password","matt");
    
    expect(await service.loginUser(loginDto)).toEqual({
      message: "The user with the specified username does not exist.",
      success: false
    });;
  });

  it('getUserByUsername-false : jwtToken not found', async () => {
    const testUsername = "ethan";
    const testToken = "xxxxx.yyyyy.zzzzz";
    
    expect(await service.getUserByUsername(testToken,testUsername)).toEqual({
      message: "This JWTToken does not exist.",
      success: false,
      user: null
    });;
  });

  it('getUserByJWTToken-false : jwtToken not found', async () => {
    const testToken = "xxxxx.yyyyy.zzzzz";
    
    expect(await service.getUserByJWTToken(testToken)).toEqual({
      message: "This JWTToken does not exist.",
      success: false,
      user: null
    });;
  });

  it('deleteUserbyUsername-false : jwtToken not found', async () => {
    const testToken = "xxxxx.yyyyy.zzzzz";
    const testUsername = "ethan";
    
    expect(await service.deleteUserByUsername(testToken, testUsername)).toEqual({
      message: "This JWTToken does not exist.",
      success: false,
    });;
  });


  test('UserResponse', async () => {
    const obj = new UserResponse(true,"test")
    expect(obj).toEqual({
      message: "test",
      success: true,
    });
  
  });

  //JWT preconditions
  it('JWT token precondition', async () => {
    const testUsername = "test";
    const testToken = "";
    
    expect(await service.getUserByUsername(testToken,testUsername)).toEqual({
      message: "Please provide a valid JWTToken.",
      success: false,
      user: null
    });;
  });
})
