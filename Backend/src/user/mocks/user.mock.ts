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
    })
  }