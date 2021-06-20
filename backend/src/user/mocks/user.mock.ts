export const MockUserService = {
    registerUser: jest.fn((dto) => {
      return {
        id: Date.now(),
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
    getUserById:jest.fn((id) => {
      const out = id + " Found";
      return out
    }),
    deleteUserById:jest.fn((id) => {
      const out = id + " Deleted";
      return out
    }),
  }
