import { RegisterUserDto } from '../dto/register-user.dto';
import { GetUserByUsernameResponse } from '../dto/get-user-by-username.dto';

export const MockUserService = {
    registerUser: jest.fn((dto) => {
      return {
        ...dto
      }
    }),
    updateUserWithId: jest.fn((id, dto) => {
      return {
        id,
        ...dto
      }
    }),
    
    getAllUsers: jest.fn(() => {
      return "User List"
    }),

    getUserByUsername:jest.fn((token,username) => {
      let resp = new GetUserByUsernameResponse(true,username,token)
      return resp;
    }),

    deleteUserById:jest.fn((id) => {
      const out = id + " Deleted";
      return out
    }),
  }
