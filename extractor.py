from preprocess import *

if __name__ == '__main__':
    itemName = "bottle"
    imgRootPath = "./images"
    extractor = ImageExtractor(imgRootPath, itemName)
    # 비디오 전처리
    extractor.preprocessVideo(SHOW_IMAGE = False)
    # 통계량 추출
    extractor.getStatistics(SHOW_PLOT=False)
    # # 결과 이미지 영역 크롭
    extractor.extractImages(SHOW_IMAGE=False)
    # print('[]', extractor.mergedAreas)