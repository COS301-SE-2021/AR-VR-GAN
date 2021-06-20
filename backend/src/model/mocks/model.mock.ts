export const MockModelService = {
    handleCoords: jest.fn((dto) => {
        let sum = 0;

        for (let i = 0; i < dto.data.length; i++) {
            sum += dto.data[i]
        }
        return sum;
    })
  }
